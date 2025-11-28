"""
@author: Carl Toller Melén

"""
# %%
# Import required packages
import numpy as np
from numpy.typing import NDArray

import gabse


# %%
# Context class for managing agents within a defined space

class Context:
    """
    A class representing the simulation context, managing agents within a defined space.

    Attributes:
    dimensions: NDArray[np.float64]
        The dimensions of the simulation environment.
    agents: list
        A list of agents present in the simulation.

    Methods:
    add_agent(agent: gabse.Agent)
        Adds a new agent to the context.
    remove_agent(agent: gabse.Agent)
        Removes an agent from the context.
    get_agents() -> list
        Returns the list of agents in the context.
    get_agents_by_class(class_name) -> list
        Returns a list of agents filtered by class name.
    get_dimensions() -> NDArray[np.float64]
        Returns the dimensions of the simulation environment.
    get_positions_array() -> NDArray[np.float64]
        Returns a (n, dim) numpy array of agent positions.
    get_agent_count(classes=None) -> dict
        Returns a dictionary with counts of agents by class name.
    """
    # Initializes the context with dimensions and empty agent list
    def __init__(self, dimensions:NDArray[np.float64]):
        self._positions_cache = None
        self.dimensions = dimensions
        self.agents = list()
        self._dirty = True

    # Adds a new agent to the list
    def add_agent(self, agent:gabse.Agent):
        self.agents.append(agent)
        self._dirty = True

    # Removes an agent from the list
    def remove_agent(self, agent:gabse.Agent):
        # Finds the right agent in the list and removes it
        self.agents.remove(agent)
        self._dirty = True

    def mark_dirty(self):
        self._dirty = True

    # Checks if an object is of a specific class name
    @staticmethod
    def check_class(obj, name) -> bool:
        if obj.__class__.__name__ == name:
            return True
        else:
            return False

    # Getters
    def get_agents(self) -> list:
        return self.agents

    def get_agents_by_class(self, class_name) -> list:
        return [agent for agent in self.agents if self.check_class(agent, class_name)]

    def get_dimensions(self) -> NDArray[np.float64]:
        return self.dimensions

    def get_positions_array(self):
        """
                Returns a (n, dim) numpy array of agent positions.
                Cached until an agent calls context.mark_dirty().
                """
        if self._dirty:
            if not self.agents:
                self._positions_cache = np.empty((0, self.dimensions.size // 2))
            else:
                # list comprehension into vstack once per rebuild
                self._positions_cache = np.vstack([a.get_position() for a in self.agents])
            self._dirty = False
        return self._positions_cache

    def get_agent_count(self, classes=None) -> dict:
        entry = dict()

        # if no classes provided, return total count for each agent type
        if not classes:
            unique_classes = set(obj.__class__.__name__ for obj in self.agents)
            for cls in unique_classes:
                a = sum(self.check_class(obj, cls) for obj in self.agents)
                entry[cls] = a
            return entry


        for arg in classes:
            # print(arg)
            a = sum(self.check_class(obj, arg) for obj in self.agents)
            # print(a)
            entry[arg] = a

        #print(entry)
        return entry

