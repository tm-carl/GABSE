"""
@author: Carl Toller Melén

"""
from typing import Any

# %%
# Import required packages
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from scipy.spatial import cKDTree as _cKDTree

import gabse


# %%
# Agent class for representing entities in the simulation

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
    position : NDArray[np.float64]
        The 3D position of the agent in the simulation space.

    Attributes
    ----------
    id: int
        Unique identifier for the agent.
    position: np.ndarray
        The 3D position of the agent in the simulation space.
    engine: Engine
        Reference to the simulation engine.
    sensor: Sensor
        The sensor associated with the agent.
    """
    # Static variable to keep track of agent IDs
    _id_counter = 0

    # Initialize agent with unique ID, position, engine reference, and empty sensor
    def __init__(self, engine:gabse.Engine, position:NDArray[np.float64]=None):
        Agent._id_counter += 1
        self.id = Agent._id_counter
        if position is None:
            position = np.array([0, 0, 0])
        self.engine = engine
        self.position = position
        self.sensor = None

    # Find nearest neighbours based on Euclidean distance
    def find_neighbours(self, agents:list, noOfNeighbours:int) -> list:
        """
        Calculates the distance between *self* and a list of *agents*, neighbours, based on Euclidean distance. It then
        filters out based on the number of neighbours to include.

        Parameters
        ----------
        agents : list
            A list of agents for which to calculate distance with.
        noOfNeighbours : int
            The number of closest neighbours to include.

        Returns
        -------
        neighbours : list
            A list of nearest agents, or single agent if *noOfNeighbours == 1*
        """
        if not agents:
            return [] if noOfNeighbours != 1 else None

        n = len(agents)
        k = min(noOfNeighbours, n)

        # Try KDTree for large n or repeated queries
        try:
            pos = np.vstack([a.get_position() for a in agents])
            tree = _cKDTree(pos)
            dists, idxs = tree.query(self.get_position(), k=k)
            if k == 1:
                return agents[int(idxs)]
            if np.isscalar(idxs):
                idxs = [int(idxs)]
            else:
                idxs = [int(i) for i in np.atleast_1d(idxs)]
            return [agents[i] for i in idxs]
        except Exception:
            # Get self position
            self_pos = self.get_position()

            # stack positions (shape: (n, dim)) and compute squared distances
            pos = self.engine.context.get_positions_array()

            if pos.size == 0:
                return [] if noOfNeighbours != 1 else None

            # compute squared Euclidean distances
            d2 = np.sum((pos - self_pos) ** 2, axis=1)

            # Return based if only one neighbours requested
            if k == 1:
                return agents[int(np.argmin(d2))]
            if k < n:
                idx_k = np.argpartition(d2, k - 1)[:k]
                idx_sorted = idx_k[np.argsort(d2[idx_k])]
            else:
                idx_sorted = np.argsort(d2)

            return [agents[i] for i in idx_sorted[:k]]


    # Check if agent is out of bounds and change the position so that it is within bounds
    def check_out_of_bounds(self) -> NDArray[np.float64]:
        """
        Checks if the agent is outside the simulation context and if so moves it to the closest point within the context.

        Returns
        -------
        position : NDArray[np.float64]
            The new position, unchanged if original position is within bounds.
        """
        bounds = np.array(self.engine.context.get_dimensions())

        minValues = bounds[0:3]
        maxValues = bounds[3:]

        return np.clip(self.position, minValues, maxValues)

    #Move agent to a specific position
    def move_position(self, position:NDArray[np.float64]):
        """
        Moves the agent to a new position. It also does a check so that the agent is still within the bounds of the context.

        Parameters
        ----------
        position : NDArray[np.float64]
            The new position where the agent it to be placed.
        """
        self.position = position
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    def move_vector(self, vector:NDArray[np.float64]):
        """
        Moves the agent to a new position based on a move vector. It also does a check so that the agent is still
        within the bounds of the context.

        Parameters
        ----------
        vector : NDArray[np.float64]
            The movement vector.
        """
        self.position += vector
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    # Calculate Euclidean distance between two agents
    def get_distance(self, agent2:"Agent") -> floating[Any]:
        """
        Calculates the Euclidean distance between two points.

        Parameters
        ----------
        self : Agent
            The first point.
        agent2 : Agent
            The second point.

        Returns
        -------
        dist : floating[Any]
            The distance
        """
        return np.linalg.norm(self.get_position() - agent2.get_position())


    def find_shortest_path(self, network):
        pass

    # Add sensor to agent
    def add_sensor(self, sensor):
        self.sensor = sensor

    # Getters and Setters
    def get_sensor(self):
        return self.sensor

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position


