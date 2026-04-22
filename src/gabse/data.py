"""
This module contains the operational data classes.
"""

# %%
# Import required packages
import numpy as np
import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import Engine
    from .agent import Agent
    from .context import Context


# %%
class Sensor:
    """
    A class representing a sensor that logs data from an agent over time. The sensor logs the sensory data based on the
    getter list that is fed as arguments when the sensor is added to the schedule.

    Parameters
    ----------
    engine: Engine
        Reference to the simulation engine.
    parent : Agent
        The agent to which the sensor is attached.
    frequency : float
        The frequency at which the sensor logs data.

    Attributes
    ----------
    engine: Engine
        Reference to the simulation engine.
    parent : Agent
        The agent to which the sensor is attached.
    logger : dict
        A dictionary to store logged data entries with the tick as the key and the data entry as the value.
    frequency : float
        The frequency at which the sensor logs data.
    """

    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, engine: "Engine", parent: "Agent | Context", frequency: float):
        self.engine = engine
        self.parent = parent
        self.logger = dict()
        self.frequency = frequency

    # Logs data entries based on specified getters
    def entry(self, *getters: list):
        """
        Logs a data entry by calling specified getter methods from the parent agent.

        Parameters
        ----------
        getters : list
            A list of names of all the getter method to call.
        """
        entry = dict() #{"tick": self.engine.schedule.tick}

        for arg in getters:
            data = getattr(self.parent, arg)
            #print(data)
            # check if data is numpy array and convert to list
            if isinstance(data, np.ndarray):
                data = (data.tolist())  # to avoid reference issues with mutable data types
            else:
                data = copy.copy(data)  # to avoid reference issues with mutable data types


            entry[arg] = data

        self.logger[self.engine.tick] = entry
        # print(self.engine.getTick())

    def merge_logger(self, other_logger: dict):
        """
        Merges another logger into this sensor's logger and sorts the combined log by tick.

        Parameters
        ----------
        other_logger : dict
            The logger to be merged.
        """
        self.logger |= other_logger

        # Sort the logger by tick to maintain chronological order
        self.logger = dict(sorted(self.logger.items()))

# %%
class DataCollector:
    """
    The data collection manager used for collecting and exporting the operational data for a simulation. The export is
    stored in a dictionary.

    Parameters
    ----------
    engine:Engine
        The simulation engine

    Attributes
    ----------
    engine:Engine
        The simulation engine
    repo:dict
        The data repository.
    """

    def __init__(self, engine):
        self.engine = engine
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

        self.repo[f"{agent.__class__.__name__} {agent.id}"] = (
            agent.sensor.logger
        )

    def collect_data(self):
        """
        Collects data from all agents' sensors and stores it in the repository. This method iterates through all
        agents in the simulation context, retrieves their logs from their sensors, and stores them in the repository
        with a key that combines the agent's class name and ID.

        Noteworthy, the collection collects logs from agents listed in the context, but does not collect logs from
        the context itself. If there is a wish to store contextual data as a log, then the recommended way is to create
        a context logger agent that collects the contextual data and stores it in its sensor log.
        """

        for agt in self.engine.context.agents:
            if agt.sensor is not None:
                self.repo[f"{agt.__class__.__name__} {agt.id}"] = (
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

    def collect_kpis(self):
        """
        Collects a key performance indicators (KPIs) and stores it in the KPI repository.
        """

        # Gets the total model time
        self.kpi["model_time"] = self.engine.tick

        c = self.engine.context
        c_kpi = dict()
        # Search for "get_kpis" method in the context and call it if exists
        method = getattr(c, "get_kpis", None)
        if callable(method):
            kpis = method()
            if isinstance(kpis, dict):
                c_kpi |= kpis
            else:
                print("get_kpis method found in context but did not return a dictionary.")


        self.kpi |= c_kpi

        # Search for "get_kpis" method in the agents and call it if exists
        for agt in self.engine.context.agents:
            agt_kpi = dict()
            method = getattr(agt, "get_kpis", None)
            if callable(method):
                kpis = method()
                if isinstance(kpis, dict):
                    agt_kpi |= kpis
                else:
                    print(f"get_kpis method found in {agt} but did not return a dictionary.")

                self.kpi[f"{agt.__class__.__name__} {agt.id}"] = agt_kpi




    def export_kpis(self):
        """
        Exports the collected KPIs.

        Returns
        -------
        kpi:dict
            The KPI repository.

        """
        return self.kpi
