"""
Copyright (C), 2025, Carl Toller Melén

This is the GABSE (Generic Agent-Based Simulation Engine) framework.

"""

# version number
__name__ = "gabse"
__author__ = "Carl Toller Melén"
__version__ = "0.1.11"
__email__ = "carl@tollermelen.se"
__status__ = "Development"

from .engine import Engine
from .agent import Agent, Sensor
from .schedule import Action, Schedule
from .context import Context
from .data import DataCollector
from .visualize import Visualizer

__all__ = [
    "Engine",
    "Action",
    "Schedule",
    "Agent",
    "Sensor",
    "Context",
    "DataCollector",
    "Visualizer"
]
