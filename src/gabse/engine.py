"""
This module contains the simulation engine class.
"""

# %%
# Import required packages
import numpy as np
from src.gabse.context import Context
from src.gabse.schedule import Schedule
from numpy.typing import NDArray


# %%
class Engine:
    """
    A class for managing the simulation engine. The engine is the main executor for the simulation and container for the
    context. The simulation is executed using the *run()* method.

    Parameters
    ----------
    modelTime : float
        The total time for which the simulation will run.
    dimensions : NDArray[np.float64]
        The dimensions of the simulation environment, based on 3D representation. The order of XYZ boundaries is done
        the following: *[X-min, Y-min, Z-min, X-max, Y-max, Z-max]*
    context : Context, optional
        The context to be used, if custom. Default is to use the built-in context.

    Attributes
    ----------
    tick : float
        The current simulation tick.
    modelTime : float
        The total time for which the simulation will run.
    dimensions : NDArray[np.float64]
        The dimensions of the simulation environment.
    context : Context
        The context containing the agents and environment of the simulation.
    schedule : Schedule
        The schedule managing the actions to be executed.
    """

    def __init__(
        self, modelTime: float, dimensions: NDArray[np.float64], context: Context = None
    ):
        self.tick = 0.0
        self.modelTime = modelTime
        self.schedule = Schedule(self.tick)
        self.dimensions = dimensions

        # Initialize context, allowing for custom context to be passed
        if context is None:
            self.context = Context(dimensions)
        else:
            self.context = context

    def run(self):
        """
        Runs the simulation until reached model time or schedule is empty
        """
        while self.tick <= self.modelTime and self.schedule.get_size() > 0:
            self.tick = self.schedule.step()
            # print(self.tick)

        # print("RUN COMPLETED!")

    def abort(self):
        """
        Aborts the simulation and prints the stopped time.
        """
        self.schedule.clear_schedule()
        print(f"Stopped at: {self.tick}")

    def get_context(self) -> Context:
        """
        Get the context connected to the simulation.

        Returns
        -------
        context : Context
            The simulation context.
        """
        return self.context
