"""
This module contains the simulation agent class.
"""

from typing import Any, Sequence, List, Iterable

# %%
# Import required packages
import numpy as np
import nanoid
from .data import Sensor
from numpy import floating
from numpy.typing import NDArray
from scipy.spatial import cKDTree as _cKDTree


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine


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

    # Static variable to keep track of agent IDs
    #_id_counter = 0

    # Initialize agent with unique ID, position, engine reference, and empty sensor
    def __init__(self,
                 engine: "Engine",
                 agent_id: str | None = None,
                 position: NDArray[np.float64] | None = None,
                 orientation: NDArray[np.float64] | None = None,
                 sensor: Sensor = None
                 ):
        #Agent._id_counter += 1
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

    def find_neighbours(self, agents: Iterable["Agent"], noOfNeighbours: int) -> list | Any:
        """
        Calculates the distance between *self* and a list of *agents*, neighbors, based on Euclidean distance. It then
        filters out based on the number of neighbors to include, minimum one.

        Parameters
        ----------
        agents : Iterable
            A list of agents for which to calculate distance with.
        noOfNeighbours : int
            The number of closest neighbors to include.

        Returns
        -------
        neighbours : list or Any
            A list of nearest agents, or single agent if *noOfNeighbours == 1*
        """
        if not agents:
            return [] if noOfNeighbours != 1 else None

        n = len(agents)
        k = min(noOfNeighbours, n)

        # Try KDTree for large n or repeated queries
        pos = np.vstack([a.position for a in agents])
        tree = _cKDTree(pos)
        dists, idxs = tree.query(self.position, k=k)
        if k == 1:
            return agents[int(idxs)]

        if np.isscalar(idxs):
            idxs = [int(idxs)]
        else:
            idxs = [int(i) for i in np.atleast_1d(idxs)]

        return [agents[i] for i in idxs]


    def check_out_of_bounds(self) -> NDArray[np.float64]:
        """
        Checks if the agent is outside the simulation context and if so moves it to the closest point within the context.

        Returns
        -------
        position : NDArray[np.float64]
            The new position, unchanged if original position is within bounds.
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
        # print(self.position)

        if orientation is not None:
            self.orientation = orientation


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


    # Calculate Euclidean distance between two agents
    def calculate_distance(self, agent2: "Agent") -> floating[Any]:
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
        return np.linalg.norm(self.position - agent2.position)