"""
@author: Carl Toller Melén

"""

# %%
# Import required packages
import numpy as np
import copy

import gabse


# %%
# Sensor class for logging agent data over time

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
    logger : list
        A list to store logged data entries.
    frequency : float
        The frequency at which the sensor logs data.

    Methods
    -------
    entry(*getters: list)
        Logs a data entry based on specified getters from the parent agent.
    get_frequency() -> float
        Returns the logging frequency of the sensor.
    get_logger() -> list
        Returns the logged data entries.
    """
    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, engine:gabse.Engine, parent:gabse.Agent, frequency:float):
        self.engine = engine
        self.parent = parent
        self.logger = list()
        self.frequency = frequency

    # Logs data entries based on specified getters
    def entry(self, *getters:list):
        """
        Logs a data entry by calling specified getter methods from the parent agent.
        """
        entry = {"tick": self.engine.get_tick()}

        for arg in getters:
            # print(g)
            method = getattr(self.parent, "get_" + arg)
            if not callable(method):
                # Tests if it is a boolean getter
                method = getattr(self.parent, "is_" + arg)
                if not callable(method):
                    continue

            data = method()
            #check if data is numpy array and convert to list
            if isinstance(data, np.ndarray):
                data = data.tolist() # to avoid reference issues with mutable data types
            else:
                try:
                    data = copy.copy(data) # to avoid reference issues with mutable data types
                except Exception:
                    pass

            entry[arg] = data

        self.logger.append(entry)
        # print(self.engine.getTick())

    # Getters
    def get_frequency(self) -> float:
        """
        Returns the logging frequency of the sensor.

        Returns
        -------
        frequency : float
            The log frequency.
        """
        return self.frequency

    def get_logger(self) -> list:
        """
        Returns the data log associated to the sensor.

        Returns
        -------
        logger : list
            The data log.
        """
        return self.logger


# %%
# Data Collector class for gathering and exporting data from agents' and context sensors

class DataCollector:
    def __init__(self, engine):
        self.engine = engine
        self.repo = dict()

    # Collects data from all agents' sensors and stores it in the repository
    def collect_data(self):
        for agt in self.engine.context.get_agents():
            self.repo[f"{agt.__class__.__name__} {agt.id}"] = agt.get_sensor().get_logger()

        # print(self.repo)

    # Exports the collected data repository
    def export_data(self):
        return self.repo