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
    A class representing an agent in the simulation. An agent will possess a specific behavior that it executes during
    the simulation. These behaviors are expressed using methods. A few standard methods for *Agent-Based Simulation (ABS)*
    are automatically included.

    The agent class is meant to be used as a parent class, i.e., any agent type that is to be used will be a child
    class of *Agent*. The child class then inherits the *Agent* behavior to ensure that it is directly compatible
    with the simulation engine and bring in standard *ABS* agent methods.

    Parameters
    ----------
    engine : Engine
        Reference to the simulation engine.
    agent_id : str, optional
        Unique identifier for the agent. Default is to automatically generate a unique ID using nanoid with a size of 7.
    position : NDArray[np.float64], optional
        The 3D position of the agent in the simulation space. Default is [0, 0, 0].
    orientation : NDArray[np.float64], optional
        The 3D orientation of the agent in the simulation space. Default is [0, 0, 0].
    sensor : Sensor, optional
        The sensor associated with the agent. Default is None.


    Attributes
    ----------
    agent_id: str
        Unique identifier for the agent, either assigned or automatically generated using nanoid with a size 7.
    position: np.ndarray
        The 3D position of the agent in the simulation space.
    engine: Engine
        Reference to the simulation engine.
    sensor: Sensor
        The sensor associated with the agent.
    """

    _GRID_OFFSET_CACHE = {}

    # Initialize agent with unique ID, position, engine reference, and empty sensor
    def __init__(self,
                 engine: "Engine",
                 agent_id: str | None = None,
                 position: NDArray[np.float64] | None = None,
                 orientation: NDArray[np.float64] | None = None,
                 sensor: "Sensor" = None
                 ):
        # Generate a unique agent_id at instantiation time when not provided.
        # (Avoid evaluating nanoid.generate at function-definition time which would
        # produce the same default for every instance.)
        if agent_id is None:
            agent_id = nanoid.generate(size=7)

        self.agent_id = agent_id
        self.engine = engine

        # Avoid using mutable objects as default arguments. Create fresh arrays
        # per instance when position/orientation not provided.
        if position is None:
            self.position = np.array([0, 0, 0], dtype=float)
        else:
            self.position = position

        if orientation is None:
            self.orientation = np.array([0, 0, 0], dtype=float)
        else:
            self.orientation = orientation

        self.sensor = sensor

    def find_neighbours(self, agents: Sequence["Agent"], n_neighbors: int) -> list | None:
        """
        Finds the *n_neighbors* nearest agents from *agents* using Euclidean distance.
        The calling agent is automatically excluded from the candidate list so an
        agent is never returned as its own neighbor.

        Parameters
        ----------
        agents : Sequence[Agent]
            The pool of agents to search among.
        n_neighbors : int
            The number of closest neighbors to return.

        Returns
        -------
        neighbours : list
            A list of nearest agents, also if *n_neighbors == 1*.
        """

        # Exclude self so the calling agent is never its own neighbor
        if self in agents:
            agents = [a for a in agents if a is not self]

        if not agents:
            return None

        n = len(agents)
        k = min(n_neighbors, n)

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

    def find_grid_neighbours(self, search_boundary: float = 1.0) -> list:
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
        neighbor_agents : list or Agent
            A list of nearest agents within the specified boundary.

        """

        cell = self.engine.context.agent_grid_cells[self.agent_id]
        grid = self.engine.context.grid
        cx, cy, cz = cell

        radius = int(search_boundary)

        # Check of grid offset already exist, otherwise generate
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

        return neighbor_agents

    def check_out_of_bounds(self) -> NDArray[np.float64]:
        """
        Clamps the agent's position to the simulation context boundaries and returns the result.

        Expects engine.context.dimensions to be a 6-element array in the form
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
        Moves the agent to a new position and orientation, optional. It also does a check so that the agent
        is still within the bounds of the context.

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

        self.engine.context.update_agent_grid(self)

    def move_vector(self, move_vector: NDArray[np.float64], rotation_vector: NDArray[np.float64] = None):
        """
        Moves and rotates the agent to a new position based on a move vector and a rotation vector, optional.
        It also does a check so that the agent is still within the bounds of the context.

        Parameters
        ----------
        move_vector : NDArray[np.float64]
            The movement vector.
        rotation_vector : NDArray[np.float64], optional
            The rotation vector.
        """
        self.position += move_vector
        self.position = self.check_out_of_bounds()
        # print(self.position)

        if rotation_vector is not None:
            self.orientation += rotation_vector

        self.engine.context.update_agent_grid(self)


    def calculate_distance(self, other_agent: "Agent") -> floating[Any]:
        """
        Calculates the Euclidean distance between this agent and *other_agent*.

        Parameters
        ----------
        other_agent : Agent
            The agent to measure the distance to.

        Returns
        -------
        dist : floating[Any]
            The Euclidean distance between the two agents.
        """
        return np.linalg.norm(self.position - other_agent.position)

# %%
class Sensor:
    """
    A class representing a sensor that logs data from an agent over time. The sensor logs the sensory data based on the
    getter list that is fed as arguments when the sensor is added to the run_schedule.

    Parameters
    ----------
    parent : Agent
        The agent to which the sensor is attached.
    frequency : float
        The frequency at which the sensor logs data.

    Attributes
    ----------
    parent : Agent | Context
        The agent or context to which the sensor is attached.
    logger : dict
        A dictionary to store logged data entries with the tick as the key and the data entry as the value.
    frequency : float
        The frequency at which the sensor logs data.
    """

    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, parent: "Agent", frequency: float):
        self.parent = parent
        self.logger = dict()
        self.frequency = frequency

    # Logs data entries based on specified getters
    def entry(self, *getters: list):
        """
        Logs a data entry by calling specified getter methods from the parent agent.

        Parameters
        ----------
        getters : list
            A list of names of all the getter method to call.
        """
        entry = dict()

        for arg in getters:
            data = getattr(self.parent, arg)

            # check if data is numpy array and convert to list
            if isinstance(data, np.ndarray):
                data = (data.tolist())  # to avoid reference issues with mutable data types
            else:
                data = copy.copy(data)  # to avoid reference issues with mutable data types


            entry[arg] = data

        self.logger[self.parent.engine.tick] = entry
        # print(self.engine.getTick())

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
