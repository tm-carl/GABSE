"""
This module contains the operational data classes.
"""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .context import Context

# %%
class DataCollector:
    """
    The data collection manager is used for collecting and exporting the operational data from a simulation. It collects both the logs from the agents' sensors and calls the key performance indicator (KPI) method from the simulation context and agents.

    Attributes
    ----------
    repo:dict
        The data repository for sensor logs.

    kpi:dict
        The key performance indicators (KPIs) repository.
    """

    def __init__(self):
        self.repo = dict()
        self.kpi = dict()

    def store_log(self, agent):
        """
        Stores the entire sensor log of a specific agent in the repository.

        Parameters
        ----------
        agent : Agent
            The agent whose sensor log is to be stored.
        """

        self.repo[f"{agent.__class__.__name__} {agent.agent_id}"] = (
            copy.copy(agent.sensor.logger) # Copy the sensor log to avoid reference issues
        )

    # ----------------------
    # Collecting and exporting results (called by Engine.run)
    # ----------------------

    def collect_data(self, agents: dict[str, "Agent"]):
        """
        Collects data from all agents' sensors and stores it in the repository. This method iterates through all agents in the simulation context. Each agent's sensor log is copied and stored in the repository with a key that combines the agent's class name and ID.

        Noteworthy, the collection collects logs from agents listed in the context, but does not collect logs from the context itself. If there is a wish to store contextual data as a log, then the recommended way is to create a *logger agent* that collects the contextual data and stores it in its sensor log.

        Parameters
        ----------
        agents : dict[str, Agent]
            The agents in the simulation, used for collecting logs from each agent's sensor if it exists. Same structure as the *Context.agents* attribute.
        """

        for agt in agents.values():
            if agt.sensor is not None:
                # Copy the sensor log to avoid reference issues
                self.repo[f"{agt.__class__.__name__} {agt.agent_id}"] = copy.copy(agt.sensor.logger)

    def export_data(self):
        """
        Exports the collected data repository.

        Returns
        -------
        repo:dict
            The data repository.

        """
        return self.repo

    def collect_kpis(self, tick: float, context: "Context", agents: dict[str, "Agent"]):
        """
        Collects a key performance indicators (KPIs) and stores it in the KPI repository. KPIs are collected from the simulation context and each agent if they have a ``get_kpis`` method defined. The collected KPIs are stored in the KPI repository with keys that combine the class name and ID of the context or agent.

        Parameters
        ----------
        tick : float
            The current simulation tick, used for storing the model time KPI.
        context : Context
            The simulation context, used for collecting KPIs from the context if a "get_kpis" method is defined.
        agents : dict[str, Agent]
            The agents in the simulation, used for collecting KPIs from each agent if a "get_kpis" method is defined in the agent class. Same structure as the *Context.agents* attribute.

        Raises
        ------
        ValueError
            If the "get_kpis" method in the context or any agent does not return a dictionary.
        """

        # Gets the total model time
        self.kpi["model_time"] = tick

        # Search for "get_kpis" method in the context and call it if exists
        context_kpis = self._read_kpis(context, "context")

        if context_kpis is not None:
            self.kpi |= context_kpis

        # Search for "get_kpis" method in the agents and call it if exists
        for agt in agents.values():
            agent_kpis = self._read_kpis(agt, agt)

            if agent_kpis is not None:
                # Store a copy, so that later changes to the dictionary returned by the agent do not alter the KPIs
                self.kpi[f"{agt.__class__.__name__} {agt.agent_id}"] = dict(agent_kpis)

    def export_kpis(self):
        """
        Exports the collected KPIs.

        Returns
        -------
        kpi:dict
            The KPI repository.

        """
        return self.kpi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_kpis(source, label) -> dict | None:
        """
        Calls the ``get_kpis`` method of *source* if it has one.

        Parameters
        ----------
        source : Context | Agent
            The context or agent to read the KPIs from.
        label : object
            What to call *source* in the error message, e.g. "context" or the agent.

        Returns
        -------
        kpis : dict or None
            The dictionary returned by ``get_kpis``, or None if *source* has no such method.

        Raises
        ------
        ValueError
            If ``get_kpis`` does not return a dictionary.
        """
        method = getattr(source, "get_kpis", None)

        if not callable(method):
            return None

        kpis = method()

        if not isinstance(kpis, dict):
            raise ValueError(f"``get_kpis`` method found in {label} but did not return a dictionary.")

        return kpis
