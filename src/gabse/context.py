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
        self.dimensions = dimensions
        self.grid = None
        #self.agents: list[Agent] = []
        self.agents: dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the context.

        Parameters
        ----------
        agent : Agent
            The agent to add.
        """
        #self.agents.append(agent)
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent: Agent):
        """
        Removes a specified agent from the context.

        Parameters
        ----------
        agent : Agent
            The agent to be removed.
        """
        #self.agents.remove(agent)
        self.agents.pop(agent.agent_id, None)

    def get_agents_by_class(self, cls: type) -> list:
        """
        Gets all agents that are instances of *cls* (including subclasses).

        Parameters
        ----------
        cls : type
            The class to filter by, e.g. ``HumanAgent``.

        Returns
        -------
        agents : list
            A list of matching agents.
        """
        return [agent for agent in self.agents.values() if isinstance(agent, cls)]

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

        return self.agents.get(agent_id)


    def get_agent_count(self, classes: list[type] | None = None) -> dict:
        """
        Gets the agent count per agent type.

        Parameters
        ----------
        classes : list[type], optional
            A list of types to count, e.g. ``[HumanAgent, ZombieAgent]``.
            If *None*, every distinct type present in the context is counted.

        Returns
        -------
        count : dict
            A dictionary mapping class name strings to their agent counts.
        """
        count: dict[str, int] = {}

        if not classes:
            unique_types = set(type(obj) for obj in self.agents.values())
            for cls in unique_types:
                count[cls.__name__] = sum(isinstance(obj, cls) for obj in self.agents.values())
            return count

        for cls in classes:
            count[cls.__name__] = sum(isinstance(obj, cls) for obj in self.agents.values())

        return count