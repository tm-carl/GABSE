"""
This module contains the simulation context class.
"""
from collections import defaultdict

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
    dimensions : NDArray[np.float64], optional
        The dimensions of the simulation environment, based on 3D representation. If no dimensions are provided,
        it will use an unbonded set (-Inf to Inf). The order of XYZ boundaries is done
        the following: *[X-min, Y-min, Z-min, X-max, Y-max, Z-max]*

    Attributes
    ----------
    dimensions: NDArray[np.float64]
        The dimensions of the simulation environment.
    grid : dict[str, set]
        A sparse grid for tracking agent locations.
    agent_grid_cells : dict[str, set]
        A reverse look-up table to know in which cell a specific agent is.
    grid_cell_size : float
        The size of each grid cell for spatial partitioning.
    agents: dict
        A dictionary of agents present in the simulation.

    """
    # Initializes the context with dimensions and empty agent list
    def __init__(
            self,
            dimensions: NDArray[np.float64] = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf]),
            grid_cell_size: float = 1.0
            ):
        self.dimensions = dimensions
        self.grid = defaultdict(set)
        self.agent_grid_cells = {}
        self.grid_cell_size = grid_cell_size

        self.agents: dict[str, Agent] = {}

    def get_grid_cell(self, pos):
        return tuple((pos // self.grid_cell_size).astype(int))

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the context.

        Parameters
        ----------
        agent : Agent
            The agent to add.
        """
        self.agents[agent.agent_id] = agent

        cell = self.get_grid_cell(agent.position)
        self.grid[cell].add(agent.agent_id)
        self.agent_grid_cells[agent.agent_id] = cell

    def update_agent_grid(self, agent: Agent):
        """
        Updates the grid cell of an agent based on its new position.

        Parameters
        ----------
        agent : Agent
            The agent whose grid cell is to be updated.
        """
        old_cell = self.agent_grid_cells.get(agent.agent_id)
        new_cell = self.get_grid_cell(agent.position)

        #print(self.grid)

        if old_cell != new_cell:
            if agent.agent_id not in self.grid.get(old_cell, set()):
                raise ValueError(f"WARNING: Agent {agent.agent_id} not in expected cell {old_cell}")

            cell_set = self.grid.get(old_cell)

            if cell_set and agent.agent_id in cell_set:
                cell_set.remove(agent.agent_id)

            if not self.grid[old_cell]:  # Clean up empty cells
                del self.grid[old_cell]

            self.grid[new_cell].add(agent.agent_id)
            self.agent_grid_cells[agent.agent_id] = new_cell

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

        cell = self.get_grid_cell(agent.position)
        self.grid[cell].remove(agent.agent_id)

        if not self.grid[cell]:  # Clean up empty cells
            del self.grid[cell]

        self.agent_grid_cells.pop(agent.agent_id, None)

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