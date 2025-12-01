"""
Copyright (C), 2025, Carl Toller Melén

This is the GABSE (Generic Agent-Based Simulation for Engineering) framework.

"""

# version number
__name__ = "gabse"
__author__ = "Carl Toller Melén"
__version__ = "0.1.4"

from .engine import Engine
from .agent import Agent
from .schedule import Action, Schedule
from .context import Context
from .data import Sensor, DataCollector

__all__ = [
    "Engine",
    "Action",
    "Schedule",
    "Agent",
    "Sensor",
    "Context",
    "DataCollector",
]
