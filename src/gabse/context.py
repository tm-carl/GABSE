"""
This module contains the simulation context class.
"""

# %%
# Import required packages
import numpy as np
from numpy.typing import NDArray

from src.gabse.agent import Agent


# %%
class Context:
    """
    A class representing the simulation context, managing agents within a defined space.

    Parameters
    ----------
    dimensions: NDArray[np.float64]
        The dimensions of the simulation environment.

    Attributes
    ----------
    dimensions: NDArray[np.float64]
        The dimensions of the simulation environment.
    agents: list
        A list of agents present in the simulation.

    """

    # Initializes the context with dimensions and empty agent list
    def __init__(self, dimensions: NDArray[np.float64]):
        self._positions_cache = None
        self.dimensions = dimensions
        self.agents = list()
        self._dirty = True

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the context.

        Parameters
        ----------
        agent : Agent
            The agent to add.
        """
        self.agents.append(agent)
        self._dirty = True

    def remove_agent(self, agent: Agent):
        """
        Removes a specified agent from the context.

        Parameters
        ----------
        agent : Agent
            The agent to be removed.
        """
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
        """
        Gets all agents in the context.

        Returns
        -------
        agents : list
            A list of agents.
        """
        return self.agents

    def get_agents_by_class(self, class_name: str) -> list:
        """
        Gets all agents of a specific class.

        Parameters
        ----------
        class_name : str
            The name of the class

        Returns
        -------
        agents : list
            A list of agents.
        """
        return [agent for agent in self.agents if self.check_class(agent, class_name)]

    def get_dimensions(self) -> NDArray[np.float64]:
        """
        Gets the dimensions of the context.

        Returns
        -------
        dimensions : NDAArray[np.float64]
            The dimensions.
        """
        return self.dimensions

    def get_positions_array(self):
        """
        Returns a (n, dim) numpy array of agent positions.
        Cached until an agent calls context.mark_dirty().

        Returns
        -------
        pos_array : np.ndarray
            The position array.
        """
        if self._dirty:
            if not self.agents:
                self._positions_cache = np.empty((0, self.dimensions.size // 2))
            else:
                # list comprehension into vstack once per rebuild
                self._positions_cache = np.vstack(
                    [a.get_position() for a in self.agents]
                )
            self._dirty = False
        return self._positions_cache

    def get_agent_count(self, classes: list = None) -> dict:
        """
        Gets the agent count for each agent type based on class.

        Parameters
        ----------
        classes : list, optional
            A List of classes to count. If *None*, then alla classes are counted.

        Returns
        -------
        counts : dict
            A dictionary with each agent class and their count.
        """
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

        # print(entry)
        return entry
