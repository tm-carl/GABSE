"""
This module contains the simulation agent class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

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

        n = len(agents)
        k = min(n_neighbors, n)

        # Use a KD-tree for efficient nearest neighbor search
        pos = np.vstack([a.position for a in agents])
        tree = _cKDTree(pos)
        dists, idxs = tree.query(self.position, k=k)
        if k == 1:
            result: list = [agents[int(idxs)]]
        else:
            if np.isscalar(idxs):
                idxs = [int(idxs)]
            else:
                idxs = [int(i) for i in np.atleast_1d(idxs)]

            result: list = [agents[i] for i in idxs]

        return result

    def find_grid_neighbours(self, search_boundary: float = 1.0) -> list | None:
        """
        Finds neighboring agents using the grid-based neighbor search. The calling agent is automatically excluded
        from the candidate list so an agent is never returned as its own neighbor.

        Parameters
        ----------
        search_boundary : float
            The width of the search area around the agent, in the same units as the agent's position. The
            search will include all grid cells that are within this distance from the agent's current cell.

        Returns
        -------
        neighbor_agents : list or None
            A list of nearest agents within the specified boundary, or None if no agents are found.

        """

        # Load the agent's current grid cell and the simulation grid
        cell = self.engine.context.agent_grid_cells[self.agent_id]
        grid = self.engine.context.grid
        cx, cy, cz = cell

        radius = int(search_boundary)

        # Check if grid offset already exist, otherwise generate
        if radius in self._GRID_OFFSET_CACHE:
            offsets = self._GRID_OFFSET_CACHE[radius]
        else:
            r_range = range(-radius, radius + 1)
            offsets = [
                (dx, dy, dz)
                for dx in r_range
                for dy in r_range
                for dz in r_range
            ]
            self._GRID_OFFSET_CACHE[radius] = offsets

        # Generate export list
        neighbor_agents = []
        extend = neighbor_agents.extend

        # Search the grids for agents
        for dx, dy, dz in offsets:
            cell_agents = grid.get((cx + dx, cy + dy, cz + dz))
            if cell_agents:
                extend([agent for agent in cell_agents if agent is not self])

        return neighbor_agents if neighbor_agents else None

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
        self.engine.context.update_agent_grid(self)

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
        self.engine.context.update_agent_grid(self)


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
