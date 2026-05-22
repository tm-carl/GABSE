# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:46:12 2025

@author: cat
"""

# %%
# Import required packages
import numpy as np

from src.gabse import Context


# Note: imports that depend on the project `src` directory are done inside methods.
# This avoids import errors when worker processes spawn without PYTHONPATH set.


# %%
# Builder class to set up the simulation environment


class Builder:
    def __init__(self, model_time=10000.0, person_quantity=100, person_speed=1, zombie_quantity=1, zombie_speed=1):
        # Simulation parameters
        self.model_time = model_time
        self.person_quantity = person_quantity
        self.person_speed = person_speed
        self.zombie_quantity = zombie_quantity
        self.zombie_speed = zombie_speed
        self.dimensions = np.array([-100.0, -100.0, 1.0, 100.0, 100.0, 1.0])

        # Initialize the simulation engine and context
        # Use the installed or local `gabse` package - runner sets PYTHONPATH to the repo `src` directory.
        import src.gabse as gabse

        context = Context(dimensions=np.array([-100.0, -100.0, 1.0, 100.0, 100.0, 1.0]),
                          grid_cell_size=1)

        self.engine = gabse.Engine(self.model_time, context)
        self.context = self.engine.context

        # Set up the simulation context with agents
        # Import gabse using the environment (runner inserts repo/src into sys.path)
        import gabse

        # Set up the simulation context with agents
        self.populate_context()

    # Method to set up the simulation context with agents
    def populate_context(self):
        import gabse

        low = self.dimensions[0:3]
        high = self.dimensions[3:]

        for i in range(self.person_quantity):
            startPos = np.array(
                [
                    low_entry
                    if low_entry == high_entry
                    else np.random.randint(low_entry, high_entry)
                    for low_entry, high_entry in zip(low, high)
                ],
                dtype="f",
            )

            # import Agents locally so module resolution uses the runner's sys.path setup
            import Agents

            p = Agents.Person(self.person_speed, self.engine, startPos)
            self.context.add_agent(p)

            a = gabse.Action(1, p, "run", interval=1.0)
            self.engine.schedule.schedule_action(a)

        for i in range(self.zombie_quantity):
            startPos = np.array(
                [
                    low_entry
                    if low_entry == high_entry
                    else np.random.randint(low_entry, high_entry)
                    for low_entry, high_entry in zip(low, high)
                ],
                dtype="f",
            )
            z = Agents.Zombie(self.zombie_speed, self.engine, startPos)
            self.context.add_agent(z)

            a = gabse.Action(1, z, "hunt", priority=10, interval=1.0)
            self.engine.schedule.schedule_action(a)

        log_agent = Agents.Logger(self.engine)
        self.context.add_agent(log_agent)

        # self.engine.run_schedule.printSchedule()