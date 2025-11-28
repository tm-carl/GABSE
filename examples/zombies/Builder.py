# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:46:12 2025

@author: cat
"""

# %%
# Import required packages
import gabse

from gabse import context
import Agents # Agents module containing Person and Zombie classes
import numpy as np


# %%
# Builder class to set up the simulation environment

class Builder:
    def __init__(self):
        # Simulation parameters
        self.modelTime = 10000.0
        self.personNum = 100
        self.personSpeed = 1
        self.zombieNum = 1
        self.zombieSpeed = 1
        self.dimensions = np.array([-100.0, -100.0, 1.0, 100.0, 100.0, 1.0])

        # Initialize the simulation engine and context
        self.engine = gabse.Engine(self.modelTime, self.dimensions)
        self.context = self.engine.get_context()
        self.dataLogger = gabse.DataCollector(self.engine)

        # Set up the simulation context with agents
        self.populate_context()

    # Method to set up the simulation context with agents
    def populate_context(self):

        low = self.dimensions[0:3]
        high = self.dimensions[3:]

        for i in range(self.personNum):
            startPos = np.array([
                l if l == h else np.random.randint(l,h)
                for l, h in zip(low, high)],
                dtype='f')
            
            p = Agents.Person(self.personSpeed, self.engine, startPos)
            self.context.add_agent(p)
            
            a = gabse.Action(1, p, "run", interval=1.0)
            self.engine.schedule.schedule_action(a)
        
        for i in range(self.zombieNum):
            startPos = np.array([
                l if l == h else np.random.randint(l,h)
                for l, h in zip(low, high)],
                dtype='f')
            z = Agents.Zombie(self.zombieSpeed, self.engine, startPos)
            self.context.add_agent(z)
            
            a = gabse.Action(1, z, "hunt", priority=10, interval=1.0)
            self.engine.schedule.schedule_action(a)

        l = Agents.Logger(self.engine)
        self.context.add_agent(l)

        #self.engine.schedule.printSchedule()

    # Getters
    def get_context(self):
        return self.context
