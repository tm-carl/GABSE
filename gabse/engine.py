"""
@author: Carl Toller Melén

"""

# %%
# Import required packages
from gabse.schedule import Schedule
from gabse.context import Context

# %%
# Engine class for managing the simulation

class Engine:
    def __init__(self, modelTime, dimensions, context=None):
        self.tick = 0.0
        self.modelTime = modelTime
        self.schedule = Schedule(self.tick)

        # Initialize context, allowing for custom context to be passed
        if context is None:
            self.context = Context(dimensions)
        else:
            self.context = context

    def run(self):
        while self.tick <= self.modelTime and self.schedule.get_size() > 0:
            self.tick = self.schedule.step()
            # print(self.tick)

        # print("RUN COMPLETED!")

    def abort(self):
        self.schedule.clear_schedule()
        print(f"Stopped at: {self.tick}")

    def get_tick(self):
        return self.tick

    def get_context(self):
        return self.context
