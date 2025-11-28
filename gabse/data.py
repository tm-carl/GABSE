"""
This module contains the operational data classes.
"""

# %%
# Import required packages
import numpy as np
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gabse.engine import Engine
    from gabse.agent import Agent


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
    logger : list
        A list to store logged data entries.
    frequency : float
        The frequency at which the sensor logs data.
    """

    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, engine: "Engine", parent: "Agent", frequency: float):
        self.engine = engine
        self.parent = parent
        self.logger = list()
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
        entry = {"tick": self.engine.schedule.get_tick()}

        for arg in getters:
            # print(g)
            method = getattr(self.parent, "get_" + arg)
            if not callable(method):
                # Tests if it is a boolean getter
                method = getattr(self.parent, "is_" + arg)
                if not callable(method):
                    continue

            data = method()
            # check if data is numpy array and convert to list
            if isinstance(data, np.ndarray):
                data = (
                    data.tolist()
                )  # to avoid reference issues with mutable data types
            else:
                try:
                    data = copy.copy(
                        data
                    )  # to avoid reference issues with mutable data types
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

    def collect_data(self):
        """
        Collects data from all agents' sensors and stores it in the repository.
        """
        for agt in self.engine.context.get_agents():
            self.repo[f"{agt.__class__.__name__} {agt.id}"] = (
                agt.get_sensor().get_logger()
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
