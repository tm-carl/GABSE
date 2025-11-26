"""
@author: Carl Toller Melén

"""

# %%
# Import required packages
import numpy as np
from scipy.spatial import cKDTree as _cKDTree

# %%
# Agent class for representing entities in the simulation

class Agent:
    # Static variable to keep track of agent IDs
    _id_counter = 0

    # Initialize agent with unique ID, position, engine reference, and empty sensor
    def __init__(self, engine, position=None):
        Agent._id_counter += 1
        self.id = Agent._id_counter
        if position is None:
            position = [0, 0, 0]
        self.engine = engine
        self.position = np.array(position)
        self.neighbours = list()
        self.sensor = ""

    # Find nearest neighbours based on Euclidean distance
    def find_neighbours(self, agents, noOfNeighbours):
        """
        agents: iterable of agents to consider (can exclude self prior to call)
        returns: list of nearest agents, or single agent if noOfNeighbours == 1
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
    def check_out_of_bounds(self):
        bounds = np.array(self.engine.context.get_dimensions())

        minValues = bounds[0:3]
        maxValues = bounds[3:]

        return np.clip(self.position, minValues, maxValues)

    #Move agent to a specific position
    def move_position(self, position):
        self.position = position
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    def move_vector(self, vector):
        self.position += vector
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    # Calculate Euclidean distance between two agents
    @staticmethod
    def get_distance(p1, p2):
        return np.linalg.norm(p1.get_position() - p2.get_position())


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


