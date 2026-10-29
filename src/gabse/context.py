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
        The dimensions of the simulation environment, based on 3D representation. If no dimensions are provided, it will use an unbonded set (-Inf to Inf). The order of XYZ boundaries is done the following: *[X-min, Y-min, Z-min, X-max, Y-max, Z-max]*
    grid_cell_size : float, optional
        The size of each grid cell for spatial partitioning. Default is 1.0.

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
    agents: dict[str, Agent]
        A dictionary of agents present in the simulation. Agent ID is used as the key and the agent object as the value.
    retired_agents: dict[str, Agent]
        Agents that were removed with *remove_agent(agent, keep_log=True)*. They are no longer part of the simulation, but their sensor logs are included in the data returned by *Engine.run(2)*. Agent ID is used as the key and the agent object as the value.

    """
    
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
        self.retired_agents: dict[str, Agent] = {}

    # ----------------------
    # Adding, removing and looking up agents
    # -----------------------

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the context.

        Parameters
        ----------
        agent : Agent
            The agent to add.
        """
        self.agents[agent.agent_id] = agent

        # Add the agent to the grid
        self._add_to_grid_cell(agent.agent_id, self._get_grid_cell(agent.position))

    def remove_agent(self, agent: Agent, keep_log: bool = False):
        """
        Removes a specified agent from the context.

        Parameters
        ----------
        agent : Agent
            The agent to be removed.
        keep_log : bool, optional
            If True, the agent is retired instead of just dropped: its remaining scheduled actions (including its sensor readings) are cancelled, so the log ends at the removal tick, and the agent is kept in *retired_agents* so that its sensor log is still part of the data returned by *Engine.run(2)*. Its KPIs are not collected, since it is no longer part of the simulation. Default is False, which removes the agent and leaves its scheduled actions untouched.

        Notes
        -----
        It is safe to retire an agent from inside one of its own actions. The agent is registered as retired before the grid is updated, so the retirement is complete even if the grid clean-up raises an error.
        """
        
        removed = self.agents.pop(agent.agent_id, None)

        # Remove the agent from the grid
        self._remove_from_grid_cell(agent.agent_id, self._get_grid_cell(agent.position))

        # Remove the agent from the reverse look-up table
        self.agent_grid_cells.pop(agent.agent_id, None)

        # Remove the agent: stop its actions (sensor readings) so the log ends now, and keep it for the log output
        if keep_log and removed is not None:
            agent.engine.schedule.cancel_agent_actions(agent)
            self.data_collector.store_log(agent)
            return

    def retire_agent(self, agent: Agent):
        """
        Retires a specified agent from the context. This is a convenience method that calls *remove_agent(agent, keep_agent=True)*.

        Parameters
        ----------
        agent : Agent
            The agent to be retired.

        Notes
        -----
        It is safe to retire an agent from inside one of its own actions. The agent is registered as retired before the grid is updated, so the retirement is complete even if the grid clean-up raises an error.
        """
        retired = self.agents.pop(agent.agent_id, None)
        
        # Remove the agent from the grid
        self._remove_from_grid_cell(agent.agent_id, self._get_grid_cell(agent.position))

        # Remove the agent from the reverse look-up table
        self.agent_grid_cells.pop(agent.agent_id, None)

        # Retire the agent: stop its actions (sensor readings) so the log ends now, and keep it for the log output
        if retired is not None:
            agent.engine.schedule.cancel_agent_actions(agent)
            self.retired_agents[agent.agent_id] = agent
            return


    def reinstate_agent(self, agent: Agent):
        """
        Reinstates a previously retired agent back into the context.

        Parameters
        ----------
        agent : Agent
            The agent to be reinstated.

        Notes
        -----
        The agent must have been previously retired using *remove_agent(agent, keep_agent=True)*. This method will re-add the agent to the context and its grid cell, allowing it to resume its actions in the simulation.
        """
        if agent.agent_id not in self.retired_agents:
            raise ValueError(f"Agent {agent.agent_id} is not in the retired agents list.")

        self.agents[agent.agent_id] = agent
        self._add_to_grid_cell(agent.agent_id, self._get_grid_cell(agent.position))
        del self.retired_agents[agent.agent_id]

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

        return self.agents.get(agent_id, None)

    def get_agent_count(self, classes: list[type] | None = None) -> dict:
        """
        Gets the agent count per agent type.

        Parameters
        ----------
        classes : list[type], optional
            A list of types to count, e.g. ``[HumanAgent, ZombieAgent]``. If *None*, every distinct type present in the context is counted.

        Returns
        -------
        count : dict
            A dictionary mapping class name strings to their agent counts.
        """
        if not classes:
            # Every distinct type present, in the order they first appear
            classes = list(dict.fromkeys(type(obj) for obj in self.agents.values()))

        return {cls.__name__: sum(isinstance(obj, cls) for obj in self.agents.values()) for cls in classes}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_grid_cell(self, pos):
        """
        Gets the grid cell for a given position.

        Parameters
        ----------
        pos : NDArray[np.float64]
            The position for which to get the grid cell.

        Returns
        -------
        cell : tuple
            The grid cell coordinates.
        """
        return tuple((pos // self.grid_cell_size).astype(int))

    def _add_to_grid_cell(self, agent_id: str, cell: tuple):
        """Puts an agent id in a grid cell and records the cell in the reverse look-up table."""
        self.grid[cell].add(agent_id)
        self.agent_grid_cells[agent_id] = cell

    def _remove_from_grid_cell(self, agent_id: str, cell: tuple):
        """
        Takes an agent id out of a grid cell and deletes the cell if it becomes empty. The reverse look-up table is left to the caller. Raises KeyError if the id is not in the cell.
        """
        cell_agent_ids = self.grid[cell]
        cell_agent_ids.remove(agent_id)

        if not cell_agent_ids:  # Clean up empty cells
            del self.grid[cell]
            
    def _update_agent_grid(self, agent: Agent):
        """
        Updates the grid cell of an agent based on its new position. Called by *Agent.move_position()* and *Agent.move_vector()*, not by user code.

        Parameters
        ----------
        agent : Agent
            The agent whose grid cell is to be updated.
        """

        old_cell = self.agent_grid_cells.get(agent.agent_id)
        new_cell = self._get_grid_cell(agent.position)

        # Update the grid only if the agent has moved to a new cell
        if old_cell != new_cell:
            # Guard: Check if the agent is in the expected old cell before removing it
            if agent.agent_id not in self.grid.get(old_cell, ()):
                raise ValueError(f"WARNING: Agent {agent.agent_id} not in expected cell {old_cell}")

            self._remove_from_grid_cell(agent.agent_id, old_cell)
            self._add_to_grid_cell(agent.agent_id, new_cell)