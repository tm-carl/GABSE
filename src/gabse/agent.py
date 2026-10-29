"""
This module contains the simulation agent class.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence
import warnings

if TYPE_CHECKING:
    from .engine import Engine

# %%
# Import required packages
import nanoid
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from scipy.spatial import cKDTree as _cKDTree
import copy


# %%
class Agent:
    """
    A class representing an agent in the simulation. The agent's behaviors are expressed using methods which can be called during the simulation using the action schedule. For each agent in a simulation model, a child class shall be created. To simplify the process of creating agent types, a base class is provided that includes common functionality and methods.

    Parameters
    ----------
    engine : Engine
        Reference to the simulation engine.
    agent_id : str, optional
        Unique identifier for the agent. Default is to automatically generate a unique ID using *nanoid* with a size of 7.
    position : NDArray[np.float64], optional
        The 3D position of the agent in the simulation space. Default is [0, 0, 0].
    orientation : NDArray[np.float64], optional
        The 3D orientation of the agent in the simulation space. Default is [0, 0, 0].


    Attributes
    ----------
    engine: Engine
            Reference to the simulation engine.
    agent_id: str
        Unique identifier for the agent, either assigned or automatically generated using *nanoid* with a size of 7.
    position: NDArray[np.float64]
        The 3D position of the agent in the simulation space.
    orientation: NDArray[np.float64]
        The 3D orientation of the agent in the simulation space.
    sensor: Sensor
        The sensor associated with the agent. Default is None, sensors is added in child classes if needed.
    """

    # Cache for grid offsets to optimize neighbor searches
    _GRID_OFFSET_CACHE = {}

    
    def __init__(self,
                 engine: "Engine",
                 agent_id: str | None = None, # Set default to None to avoid reoccurring ID generation
                 position: NDArray[np.float64] | None = None, # Set default to None to avoid reoccurring position generation
                 orientation: NDArray[np.float64] | None = None # Set default to None to avoid reoccurring orientation generation
                 ):

        self.engine = engine

        # Generate a unique agent_id at instantiation time when not provided.
        if agent_id is None:
            self.agent_id = nanoid.generate(size=7)
        else:
            self.agent_id = agent_id

        # Sets the initial position of the agent, defaulting to [0, 0, 0] if not provided.
        if position is None:
            self.position = np.array([0, 0, 0], dtype=float)
        else:
            self.position = position

        # Sets the initial orientation of the agent, defaulting to [0, 0, 0] if not provided.
        if orientation is None:
            self.orientation = np.array([0, 0, 0], dtype=float)
        else:
            self.orientation = orientation

        # Initializes the sensor to None, indicating that the agent does not have an associated sensor by default.
        self.sensor = None

    # ----------------------
    # Neighbour search methods
    # ----------------------

    def find_neighbours(self, agents: Sequence["Agent"], n_neighbors: int) -> list | None:
        """
        Finds the *n_neighbors* nearest agents from *self* using Euclidean distance. The calling agent is automatically excluded from the candidate list so an agent is never returned as its own neighbor.

        Parameters
        ----------
        agents : Sequence[Agent]
            The pool of agents to search among.
        n_neighbors : int
            The number of closest neighbors to return.

        Returns
        -------
        neighbours : list or None
            A list of nearest agents, also if *n_neighbors == 1*. Returns None if no agents are found.
        """

        # Guard: Exclude self so the calling agent is never its own neighbor
        if self in agents:
            agents = [a for a in agents if a is not self]

        # Guard: Return None if no agents are found
        if not agents:
            return None

        k = min(n_neighbors, len(agents))

        # Use a KD-tree for efficient nearest neighbor search
        pos = np.vstack([a.position for a in agents])
        tree = _cKDTree(pos)
        _, idxs = tree.query(self.position, k=k)

        # The tree returns a single index for k == 1 and an array of indices otherwise
        return [agents[int(i)] for i in np.atleast_1d(idxs)]

    def find_grid_neighbours(self, search_boundary: float = 1.0, search_shape: str = "cubic") -> list | None:
        """
        Finds neighboring agents using the grid-based neighbor search. The calling agent is automatically excluded
        from the result so an agent is never returned as its own neighbor.

        Parameters
        ----------
        search_boundary : float
            The search distance around the agent, in the same units as the agent's position. The search covers every
            grid cell within this distance of the agent's current cell, i.e. *ceil(search_boundary / grid_cell_size)*
            cells in each direction. The search is done per cell, so an agent in a covered cell can be somewhat
            further away than *search_boundary*. A distance of 0 searches the agent's own cell only.
        search_shape : str
            The shape of the search area, either "cubic" or "spherical". A cubic search covers every cell in the
            (2r+1)^3 block around the agent's cell. A spherical search covers only the cells whose offset from the
            agent's cell is within *r* cells (Euclidean distance between cell indices), where r is
            ceil(search_boundary / grid_cell_size). The shape is applied to grid cells, not to agent positions.

        Returns
        -------
        neighbor_agents : list or None
            A list of the agents found (the agent objects, not their ids), or None if no agents are found.

        Raises
        ------
        ValueError
            The agent has not been added to the context, or *search_boundary* is negative.

        Warnings
        --------
        UserWarning
            If *search_shape* is not "cubic" or "spherical", it defaults to "cubic" and a warning is issued.
        """

        # Load the agent's current grid cell and the simulation grid
        context = self.engine.context

        if self.agent_id not in context.agent_grid_cells:
            raise ValueError(f"Agent {self.agent_id} must be added to the context before searching for neighbours.")

        if search_boundary < 0:
            raise ValueError(f"search_boundary must not be negative, got {search_boundary}.")

        if search_shape not in ("cubic", "spherical"):
            warnings.warn(f"search_shape must be either 'cubic' or 'spherical', got {search_shape}. Defulting to 'cubic'.")
            search_shape = "cubic"

        cx, cy, cz = context.agent_grid_cells[self.agent_id]
        grid = context.grid

        # Convert the distance in position units to a number of grid cells
        radius = math.ceil(search_boundary / context.grid_cell_size)

        # Check if grid offsets already exist for this radius and shape, otherwise generate
        cache_key = (radius, search_shape)
        if cache_key in self._GRID_OFFSET_CACHE:
            offsets = self._GRID_OFFSET_CACHE[cache_key]
        else:
            r_range = range(-radius, radius + 1)
            offsets = [
                (dx, dy, dz)
                for dx in r_range
                for dy in r_range
                for dz in r_range
            ]
            if search_shape == "spherical":
                r_sq = radius * radius
                offsets = [
                    (dx, dy, dz) for dx, dy, dz in offsets
                    if dx * dx + dy * dy + dz * dz <= r_sq
                ]
            self._GRID_OFFSET_CACHE[cache_key] = offsets

        # Search the grids for agents. The grid holds agent ids, so each id is looked up in the context; an id that
        # is no longer registered there (a stale grid entry) is skipped.
        neighbor_agents = []
        agents = context.agents

        for dx, dy, dz in offsets:
            cell_agent_ids = grid.get((cx + dx, cy + dy, cz + dz))
            if cell_agent_ids:
                for agent_id in cell_agent_ids:
                    if agent_id != self.agent_id and agent_id in agents:
                        neighbor_agents.append(agents[agent_id])

        return neighbor_agents if neighbor_agents else None

    # ----------------------
    # Position and movement
    # ----------------------

    def check_out_of_bounds(self) -> NDArray[np.float64]:
        """
        Clamps the agent's position to the simulation context boundaries and returns the result.

        Expects engine.context.dimensions to be a 6-element array in the following order:
        [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns
        -------
        position : NDArray[np.float64]
            The clamped position; unchanged if the agent was already within bounds.
        """
        bounds = np.array(self.engine.context.dimensions)

        minValues = bounds[0:3]
        maxValues = bounds[3:]

        return np.clip(self.position, minValues, maxValues)

    def move_position(self, position: NDArray[np.float64], orientation: NDArray[np.float64] = None):
        """
        Moves the agent to a new position and orientation, optional. It also does a check so that the agent is still within the bounds of the context.

        Parameters
        ----------
        position : NDArray[np.float64]
            The new position where the agent it to be placed.
        orientation : NDArray[np.float64], optional
            The new orientation of the agent.
        """

        self.position = position
        self.position = self.check_out_of_bounds()

        if orientation is not None:
            self.orientation = orientation

        # Update the agent's position in the context's grid after moving
        self.engine.context._update_agent_grid(self)

    def move_vector(self, move_vector: NDArray[np.float64], rotation_vector: NDArray[np.float64] = None):
        """
        Moves and rotates the agent to a new position based on a move vector and a rotation vector, optional. It also does a check so that the agent is still within the bounds of the context.

        Parameters
        ----------
        move_vector : NDArray[np.float64]
            The movement vector.
        rotation_vector : NDArray[np.float64], optional
            The rotation vector.
        """

        self.position += move_vector
        self.position = self.check_out_of_bounds()

        if rotation_vector is not None:
            self.orientation += rotation_vector

        # Update the agent's position in the context's grid after moving
        self.engine.context._update_agent_grid(self)


    def calculate_distance(self, other_agent: "Agent") -> float:
        """
        Calculates the Euclidean distance between *self* and *other_agent*.

        Parameters
        ----------
        other_agent : Agent
            The agent to measure the distance to.

        Returns
        -------
        distance : float
            The Euclidean distance between the two agents.
        """
        return float(np.linalg.norm(self.position - other_agent.position))

# %%
class Sensor:
    """
    A class representing a virtual sensor that logs data from an agent at a given frequency. The sensor logs the parent agent's properties and values based on the getter list. Frequency is determined when the sensor's *entry* method is added to the action schedule of the simulaiton, using the interval parameter. The sensor can also merge its log with another sensor's log.

    Parameters
    ----------
    parent : Agent
        The agent to which the sensor is attached.

    Attributes
    ----------
    parent : Agent | Context
        The agent or context to which the sensor is attached.
    logger : dict
        A dictionary to store logged data entries with the tick as the key and the data entry as the value.
    """

    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, parent: "Agent"):
        self.parent = parent
        self.logger = dict()

    # Logs data entries based on specified getters
    def entry(self, *getters: list):
        """
        Logs a data entry by reading properties/attributes from the parent agent.

        Parameters
        ----------
        getters : list[str]
            A list of names of all the propoerties/attributes to log. 
        """
        entry = dict()

        for arg in getters:
            data = getattr(self.parent, arg)

            # check if data is numpy array and convert to list
            if isinstance(data, np.ndarray):
                data = (data.tolist())
            else:
                data = copy.copy(data)  # to avoid reference issues with mutable data types

            # Store the data in the entry dictionary with the property name as the key
            entry[arg] = data

        # Store the entry in the logger with the current tick as the key
        self.logger[self.parent.engine.tick] = entry

    def merge_logger(self, other_logger: dict):
        """
        Merges another logger into this sensor's logger and sorts the combined log by tick. 

        Parameters
        ----------
        other_logger : dict
            The logger to be merged.
        """
        self.logger |= other_logger

        # Sort the logger by tick to maintain chronological order
        self.logger = dict(sorted(self.logger.items()))
