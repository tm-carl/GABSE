"""
Copyright (C), 2025, Carl Toller Melén

This is the GABSE (Generic Agent-Based Simulation for Engineering) framework.

"""

# version number
__name__ = "gabse"
__author__ = "Carl Toller Melén"
__version__ = "0.1.2"

from src.gabse.engine import Engine
from src.gabse.agent import Agent
from src.gabse.schedule import Action, Schedule
from src.gabse.context import Context
from src.gabse.data import Sensor, DataCollector

__all__ = [
    "Engine",
    "Action",
    "Schedule",
    "Agent",
    "Sensor",
    "Context",
    "DataCollector",
]
