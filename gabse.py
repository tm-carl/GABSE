"""
Created on Thu Oct 23 10:57:59 2025

@author: Carl Toller Melén

This is the GABSE (Generic Agent-Based Simulation for Engineering) framework. It provides classes
and methods to create and manage simulations involving agents, their actions, sensors, and the
simulation context. It is based on agent-based modeling technique and is developed with the intention
of being lightweight, scalable, and flexible. This package provides the engine, action scheduling,
generic agent functionality, and sensory data collection and management tools.

EXAMPLES:
    - Zombie apocalypse (both 2D and 3D)

REQUIRED packages:
    - sortedcontainers
    - numpy
    - copy
"""
# %%
# Import required packages

from sortedcontainers import SortedList
import numpy as np
import copy
from scipy.spatial import cKDTree as _cKDTree


# %%
# Engine class for managing the simulation

class Engine:
    def __init__(self, modelTime, dimensions):
        self.tick = 0.0
        self.modelTime = modelTime
        self.schedule = Schedule(self.tick)
        self.context = Context(dimensions)

    
    def run(self):
        while self.tick <= self.modelTime and self.schedule.get_size() > 0:
            self.tick = self.schedule.step()
            #print(self.tick)

        #print("RUN COMPLETED!")
    
    def abort(self):
        self.schedule.clear_schedule()
        print(f"Stopped at: {self.tick}")
        
    def get_tick(self):
        return self.tick

    def get_context(self):
        return self.context

# %%
# Action class for representing scheduled actions

class Action:
    """
    tick: float
        The simulation tick at which the action is scheduled to occur.
    agent: Agent
        The agent that will perform the action.
    method: str
        The name of the method to be called on the agent.
    args: list, optional
        The arguments to be passed to the method. Can be None, empty list, or "" if no arguments are needed.
    priority: int, optional
        The priority of the action (lower values indicate higher priority). Default is 0.
    interval: float, optional
        The interval for recurring actions. If greater than 0, the action will be rescheduled
        after execution. Default is 0.
    """
    def __init__(self, tick, agent, method, args=None, priority=0, interval=0):
        self.tick = float(tick)
        self.agent = agent
        self.method = method
        self.args = args
        self.priority = int(priority)
        self.interval = float(interval)

    def __str__(self):
        return f"Action entry:\ntick: {self.tick}, agent: {self.agent}, method: {self.method}, arguments: {self.args}, priority: {self.priority}, interval: {self.interval}"


# %%
# Schedule class for managing and executing scheduled actions
class Schedule:

    # Creates an empty schedule (list) and tick timer, set to zero
    # List is sorted based on tick value of actions and priority
    def __init__(self, tick):
        self.schedule = SortedList(key=lambda a: (a.tick, a.priority))
        self.tick = tick

    # Schedule method for adding an action in schedule
    def schedule_action(self, action: Action):
        self.schedule.add(action)

    # Method for stepping forward in simulation
    def step(self):
        # If schedule is empty, return current tick
        if not self.schedule:
            return self.tick

        # Checks if previous actions exist and, if so, removes them
        while self.schedule[0].tick < self.tick:
            self.schedule.pop(0)

        # If schedule is empty after removing past actions, return current tick
        if not self.schedule:
            return self.tick

        # Load the first action in schedule
        action = self.schedule[0]

        # Step to next action tick
        self.tick = action.tick

        # Calls action agent method
        method = getattr(action.agent, action.method)

        # print(f"{action.agent} at {self.tick}")
        # print(action.method)
        # print(args)

        # Check and call
        if callable(method):
            if action.args is None or len(action.args) == 0 or action.args == "":
                method()
            else:
                method(*action.args)
            # print(result)  # Output: 1, 2, 3
        else:
            print("Method not found or not callable.")

        # Checks if the action is recurring and, if so, schedules next instance
        if action.interval > 0.0:
            nextAction = Action(
                tick=action.tick + action.interval,
                agent=action.agent,
                method=action.method,
                args=action.args,
                interval=action.interval
            )
            self.schedule_action(nextAction)

        # Remove the executed action from the schedule
        self.schedule.pop(0)

        #Return current tick
        return self.tick


    # Filter out all actions related to the target agent
    def remove_agent_from_list(self, target):
        self.schedule = SortedList(
            [action for action in self.schedule if action.agent != target],
            key=lambda action: (action.tick, action.priority)
        )

    def get_schedule(self):
        return self.schedule

    def print_schedule(self):
        for action in self.schedule:
            print(action)

    def get_tick(self):
        return self.tick

    def get_size(self):
        return len(self.schedule)

    def clear_schedule(self):
        self.schedule.clear()

# %%
# Agent class for representing entities in the simulation

class Agent:
    # Static variable to keep track of agent IDs
    _id_counter = 0

    # Initialize agent with unique ID, position, engine reference, and empty sensor
    def __init__(self, engine, position=None):
        Agent._id_counter += 1
        self.id = Agent._id_counter
        if position is None:
            position = [0, 0, 0]
        self.engine = engine
        self.position = np.array(position)
        self.neighbours = list()
        self.sensor = ""

    # Find nearest neighbours based on Euclidean distance
    def find_neighbours(self, agents, noOfNeighbours):
        """
        agents: iterable of agents to consider (can exclude self prior to call)
        returns: list of nearest agents, or single agent if noOfNeighbours == 1
        """
        if not agents:
            return [] if noOfNeighbours != 1 else None

        n = len(agents)
        k = min(noOfNeighbours, n)

        # Try KDTree for large n or repeated queries
        try:
            pos = np.vstack([a.get_position() for a in agents])
            tree = _cKDTree(pos)
            dists, idxs = tree.query(self.get_position(), k=k)
            if k == 1:
                return agents[int(idxs)]
            if np.isscalar(idxs):
                idxs = [int(idxs)]
            else:
                idxs = [int(i) for i in np.atleast_1d(idxs)]
            return [agents[i] for i in idxs]
        except Exception:
            # Get self position
            self_pos = self.get_position()

            # stack positions (shape: (n, dim)) and compute squared distances
            pos = self.engine.context.get_positions_array()

            if pos.size == 0:
                return [] if noOfNeighbours != 1 else None

            # compute squared Euclidean distances
            d2 = np.sum((pos - self_pos) ** 2, axis=1)

            # Return based if only one neighbours requested
            if k == 1:
                return agents[int(np.argmin(d2))]
            if k < n:
                idx_k = np.argpartition(d2, k - 1)[:k]
                idx_sorted = idx_k[np.argsort(d2[idx_k])]
            else:
                idx_sorted = np.argsort(d2)

            return [agents[i] for i in idx_sorted[:k]]


    # Check if agent is out of bounds and change the position so that it is within bounds
    def check_out_of_bounds(self):
        bounds = np.array(self.engine.context.get_dimensions())

        minValues = bounds[0:3]
        maxValues = bounds[3:]

        return np.clip(self.position, minValues, maxValues)

    #Move agent to a specific position
    def move_position(self, position):
        self.position = position
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    def move_vector(self, vector):
        self.position += vector
        self.position = self.check_out_of_bounds()
        # print(self.position)

        try:
            self.engine.context.mark_dirty()
        except Exception:
            pass

    # Calculate Euclidean distance between two agents
    @staticmethod
    def get_distance(p1, p2):
        return np.linalg.norm(p1.get_position() - p2.get_position())


    def find_shortest_path(self, network):
        pass

    # Add sensor to agent
    def add_sensor(self, sensor):
        self.sensor = sensor

    # Getters and Setters
    def get_sensor(self):
        return self.sensor

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position



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
# Context class for managing agents within a defined space

class Context:
    # Initializes the context with dimensions and empty agent list
    def __init__(self, dimensions):
        self._positions_cache = None
        self.dimensions = np.array(dimensions)
        self.agents = list()
        self._dirty = True

    # Adds a new agent to the list
    def add_agent(self, agent):
        self.agents.append(agent)
        self._dirty = True

    # Removes an agent from the list
    def remove_agent(self, agent):
        # Finds the right agent in the list and removes it
        self.agents.remove(agent)
        self._dirty = True

    def mark_dirty(self):
        self._dirty = True

    # Checks if an object is of a specific class name
    @staticmethod
    def check_class(obj, name):
        if obj.__class__.__name__ == name:
            return True
        else:
            return False

    # Getters
    def get_agents(self):
        return self.agents

    def get_dimensions(self):
        return self.dimensions

    def get_positions_array(self):
        """
                Returns a (n, dim) numpy array of agent positions.
                Cached until an agent calls context.mark_dirty().
                """
        if self._dirty:
            if not self.agents:
                self._positions_cache = np.empty((0, self.dimensions.size // 2))
            else:
                # list comprehension into vstack once per rebuild
                self._positions_cache = np.vstack([a.get_position() for a in self.agents])
            self._dirty = False
        return self._positions_cache

    def get_agent_count(self, classes=None):
        entry = dict()

        # if no classes provided, return total count for each agent type
        if not classes:
            unique_classes = set(obj.__class__.__name__ for obj in self.agents)
            for cls in unique_classes:
                a = sum(self.check_class(obj, cls) for obj in self.agents)
                entry[cls] = a
            return entry


        for arg in classes:
            # print(arg)
            a = sum(self.check_class(obj, arg) for obj in self.agents)
            # print(a)
            entry[arg] = a

        #print(entry)
        return entry

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