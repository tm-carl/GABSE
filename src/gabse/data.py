"""
This module contains the operational data classes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .context import Context

# %%
class DataCollector:
    """
    The data collection manager used for collecting and exporting the operational data for a simulation. The export is
    stored in a dictionary.

    Attributes
    ----------
    repo:dict
        The data repository.

    kpi:dict
        The key performance indicators (KPIs) repository.
    """

    def __init__(self):
        self.repo = dict()
        self.kpi = dict()

    def store_log(self, agent):
        """
        Stores the entire log of a specific agent in the repository.

        Parameters
        ----------
        agent : Agent
            The agent whose log is to be stored.
        """

        self.repo[f"{agent.__class__.__name__} {agent.agent_id}"] = (
            agent.sensor.logger
        )

    def collect_data(self, agents: dict[str, "Agent"]):
        """
        Collects data from all agents' sensors and stores it in the repository. This method iterates through all
        agents in the simulation context, retrieves their logs from their sensors, and stores them in the repository
        with a key that combines the agent's class name and ID.

        Noteworthy, the collection collects logs from agents listed in the context, but does not collect logs from
        the context itself. If there is a wish to store contextual data as a log, then the recommended way is to create
        a context logger agent that collects the contextual data and stores it in its sensor log.

        Parameters
        ----------
        agents : dict[str, Agent]
            The agents in the simulation, used for collecting logs from each agent's sensor if it exists.
        """

        for agt in agents.values():
            if agt.sensor is not None:
                self.repo[f"{agt.__class__.__name__} {agt.agent_id}"] = (
                    agt.sensor.logger
                )

        # print(self.repo)

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
        Collects a key performance indicators (KPIs) and stores it in the KPI repository.

        Parameters
        ----------
        tick : float
            The current simulation tick, used for storing the model time KPI.
        context : Context
            The simulation context, used for collecting KPIs from the context if a "get_kpis" method is defined.
        agents : dict[str, Agent]
            The agents in the simulation, used for collecting KPIs from each agent if a "get_kpis" method is defined in the agent class.
        """

        # Gets the total model time
        self.kpi["model_time"] = tick

        c_kpi = dict()
        # Search for "get_kpis" method in the context and call it if exists
        method = getattr(context, "get_kpis", None)
        if callable(method):
            kpis = method()
            if isinstance(kpis, dict):
                c_kpi |= kpis
            else:
                print("get_kpis method found in context but did not return a dictionary.")


        self.kpi |= c_kpi

        # Search for "get_kpis" method in the agents and call it if exists
        for agt in agents.values():
            agt_kpi = dict()
            method = getattr(agt, "get_kpis", None)
            if callable(method):
                kpis = method()
                if isinstance(kpis, dict):
                    agt_kpi |= kpis
                else:
                    print(f"get_kpis method found in {agt} but did not return a dictionary.")

                self.kpi[f"{agt.__class__.__name__} {agt.agent_id}"] = agt_kpi

    def export_kpis(self):
        """
        Exports the collected KPIs.

        Returns
        -------
        kpi:dict
            The KPI repository.

        """
        return self.kpi
