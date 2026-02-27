"""
This module contains the simulation engine class.
"""


# %%
# Import required packages
import numpy as np

from .data import DataCollector
from .context import Context
from .schedule import Schedule
from numpy.typing import NDArray


# %%
class Engine:
    """
    A class for managing the simulation engine. The engine is the main executor for the simulation and container for the
    context. The simulation is executed using the *run()* method.

    Parameters
    ----------
    model_time : float
        The total time for which the simulation will run.
    dimensions : NDArray[np.float64]
        The dimensions of the simulation environment, based on 3D representation. The order of XYZ boundaries is done
        the following: *[X-min, Y-min, Z-min, X-max, Y-max, Z-max]*
    context : Context, optional
        The context to be used, if custom. Default is to use the built-in context.
    collect_data : bool, optional
        Whether to collect data at the end of the simulation. Default is False.

    Attributes
    ----------
    tick : float
        The current simulation tick.
    model_time : float
        The total time for which the simulation will run.
    dimensions : NDArray[np.float64]
        The dimensions of the simulation environment.
    context : Context | Any
        The context containing the agents and environment of the simulation. Can also be a child class of Context class.
    schedule : Schedule
        The schedule managing the actions to be executed.
    """

    def __init__(
        self,
            model_time: float,
            dimensions: NDArray[np.float64],
            context: Context = None
    ):
        self.tick = 0.0
        self.model_time = model_time
        self.schedule = Schedule(self.tick)
        self.data_logger = DataCollector(self)
        self.dimensions = dimensions

        # Initialize context, allowing for custom context to be passed
        if context is None:
            self.context = Context(dimensions)
        else:
            self.context = context

    def run(self, no_arg_out:int = 0) -> None | dict | tuple:
        """
        Runs the simulation until reached model time, schedule is empty, or simulation is aborted internally.

        Parameters
        ----------
        no_arg_out : int, optional
            The number of output arguments to return. Default is 0 (returns nothing). If 1, returns the collected KPIs
            as a dictionary. If 2, returns a tuple containing the collected KPIs and the collected data as dictionaries.

        Returns
        -------
        None | dict | tuple
            The return value depends on the value of *no_arg_out*. If *no_arg_out* is 0, the method returns `None`.
            If *no_arg_out* is 1, it returns a dictionary containing the collected KPIs. If *no_arg_out* is 2, it
            returns a tuple containing the collected KPIs and the collected data as dictionaries.
        None
            If *no_arg_out* is 0, the method returns nothing.
        KPIs : dict
            The collected KPIs from the simulation, returned if *no_arg_out* is 1 or 2.
        data : dict
            The collected data from the simulation, returned if *no_arg_out* is 2.
        """

        # Continuously steps through the schedule until the model time is reached or the schedule is empty.
        while self.tick <= self.model_time and self.schedule.get_size() > 0:
            self.tick = self.schedule.step()
            # print(self.tick)

        # Return collected KPIs and data based on the value of no_arg_out
        if no_arg_out == 0: # default, returns nothing
            return None
        elif no_arg_out == 1: # returns collected KPIs as a dictionary
            self.data_logger.collect_kpis()
            return self.data_logger.export_kpis()
        elif no_arg_out == 2: # returns collected KPIs and data as a tuple of dictionaries
            self.data_logger.collect_kpis()
            self.data_logger.collect_data()
            return self.data_logger.export_kpis(), self.data_logger.export_data()
        else: # if no_arg_out is not 0, 1, or 2, returns nothing and prints a warning message.
            print("Warning: no_arg_out should be 0, 1, or 2. Returning nothing.")
            return None
        # print("RUN COMPLETED!")

    def abort(self):
        """
        Aborts the simulation and prints the stopped time. If data collection is enabled, it will also collect the
        data up to the point of abortion.
        """
        self.schedule.clear_schedule()
        #print(f"Stopped at: {self.tick}")

    def get_context(self) -> Context:
        """
        Get the context connected to the simulation.

        Returns
        -------
        context : Context
            The simulation context.
        """
        return self.context
