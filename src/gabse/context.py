"""
This module contains the simulation context class.
"""

# %%
# Import required packages
import numpy as np
from numpy.typing import NDArray

from .agent import Agent


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
        self.grid = None
        self.agents = list()

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the context.

        Parameters
        ----------
        agent : Agent
            The agent to add.
        """
        self.agents.append(agent)

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

    # Checks if an object is of a specific class name
    @staticmethod
    def check_class(obj, name) -> bool:
        if obj.__class__.__name__ == name:
            return True
        else:
            return False

    def get_agents_positions(self):
        """
        Collects the positions of all agents in the context and returns them as a numpy array.

        Returns
        -------
        positions : NDArray[np.float64]
            A numpy array containing the positions of all agents.
        """

        repo = dict()

        for agent in self.agents:
            repo[agent.agent_id] = (agent.__class__.__name, agent.position)

        return repo

    # Getters
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

    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """
        Gets an agent by its unique identifier.

        Parameters
        ----------
        agent_id : str
            The unique identifier of the agent.

        Returns
        -------
        agent : Agent
            The agent with the specified unique identifier, or None if not found.
        """

        return next((a for a in self.agents if a.agent_id == agent_id), None)


    def get_agent_count(self, classes: list = None) -> dict:
        """
        Gets the agent count for each agent type based on class.

        Parameters
        ----------
        classes : list, optional
            A List of classes to count. If *None*, then alla classes are counted.

        Returns
        -------
        count : dict
            A dictionary with each agent class and their count.
        """
        count = dict()

        # if no classes provided, return total count for each agent type
        if not classes:
            unique_classes = set(obj.__class__.__name__ for obj in self.agents)
            for cls in unique_classes:
                a = sum(self.check_class(obj, cls) for obj in self.agents)
                count[cls] = a
            return count

        for arg in classes:
            # print(arg)
            a = sum(self.check_class(obj, arg) for obj in self.agents)
            # print(a)
            count[arg] = a

        # print(count)
        return count