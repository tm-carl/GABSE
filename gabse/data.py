"""
@author: Carl Toller Melén

"""

# %%
# Import required packages
import numpy as np
import copy

# %%
# Sensor class for logging agent data over time

class Sensor:
    # Initializes the sensor with engine reference, parent agent, empty logger, and frequency
    def __init__(self, engine, parent, frequency):
        self.engine = engine
        self.parent = parent
        self.logger = list()
        self.frequency = frequency

    # Logs data entries based on specified getters
    def entry(self, *getters):
        entry = {"tick": self.engine.get_tick()}

        for arg in getters:
            # print(g)
            method = getattr(self.parent, "get_" + arg)
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
    def get_frequency(self):
        return self.frequency

    def get_logger(self):
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