"""
This module contains the simulation engine class.
"""


# %%
# Import required packages
import numpy as np

from .data import DataCollector
from .context import Context
from .schedule import Schedule, Action
from numpy.typing import NDArray

#%%

def call_action(action: Action):
    """
    Calls the method specified in the action on the agent with the provided arguments.

    Parameters
    ----------
    action : Action
        The action to be called.
    """
    method = getattr(action.agent, action.method)

    args = []

    if action.args is not None:
        args = list(action.args)

    # Check and call
    if callable(method):
        if action.args is None or len(args) == 0:
            method()
        else:
            method(*args)
    else:
        raise ValueError("Method not found or not callable.")

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
        The run_schedule managing the actions to be executed.
    aborted : bool
        Whether the simulation is aborted or not.
    """

    def __init__(
            self,
            model_time: float,
            dimensions: NDArray[np.float64],
            context: Context = None
            ):

        self.tick = 0.0
        self.model_time = model_time
        self.schedule = Schedule()
        self.data_logger = DataCollector()
        self.dimensions = dimensions
        self.aborted = False

        # Initialize context, allowing for custom context to be passed
        if context is None:
            self.context = Context(dimensions)
        else:
            self.context = context

    def run(self, no_arg_out:int = 0) -> None | dict | tuple:
        """
        Runs the simulation until reached model time, run_schedule is empty, or simulation is aborted internally.

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

        # Continuously steps through the run_schedule until the model time is reached or the run_schedule is empty.
        while self.tick <= self.model_time and len(self.schedule.run_schedule) > 0 and not self.aborted:
            self.step(self.tick)


        # At the end of the simulation, iterate through end actions assigned (allowing for final events to occur)
        while len(self.schedule.end_schedule) > 0:
            self.end_step()

        # Return collected KPIs and data based on the value of no_arg_out
        if no_arg_out == 0: # default, returns nothing
            return None
        elif no_arg_out == 1: # returns collected KPIs as a dictionary
            self.data_logger.collect_kpis(tick=self.tick, context=self.context, agents=self.context.agents)
            return self.data_logger.export_kpis()
        elif no_arg_out == 2: # returns collected KPIs and data as a tuple of dictionaries
            self.data_logger.collect_kpis(tick=self.tick, context=self.context, agents=self.context.agents)
            self.data_logger.collect_data(agents=self.context.agents)
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
        self.schedule.run_schedule.clear()
        self.aborted = True
        #print(f"Stopped at: {self.tick}")

    def step(self, old_tick):
        """
        Steps one entry in the run_schedule. The step method executes the next action entry and, if reoccurring, re-schedules
        it. It also moves the tick forward one instance, can be the same if multiple actions are scheduled at the same tick.

        Parameters
        ----------
        old_tick : float
            The current tick before stepping.

        """
        # Guard: return early if the run_schedule is empty
        if not self.schedule.run_schedule:
            return

        # Remove any stale actions whose tick is earlier than the current engine tick
        while self.schedule.run_schedule and self.schedule.run_schedule[0].tick < old_tick:
            self.schedule.run_schedule.pop(0)

        # Guard: return early if all remaining actions were stale
        if not self.schedule.run_schedule:
            return

        # Load the first action in run_schedule
        action = self.schedule.run_schedule[0]

        # Step to next action tick and set the engine ticker to this
        self.tick = action.tick

        # Calls action agent method
        call_action(action)

        # Checks if the action is recurring and, if so, schedules next instance
        if action.interval > 0.0:
            nextAction = Action(
                tick=action.tick + action.interval,
                agent=action.agent,
                method=action.method,
                args=action.args,
                interval=action.interval,
            )
            self.schedule.schedule_action(nextAction)

        # Remove the executed action from the run_schedule
        self.schedule.run_schedule.pop(0)

    def end_step(self):
        """
        Executes the next action in the end-of-simulation queue (``end_schedule``).
        Called by the engine after the main run_schedule is exhausted or the simulation
        is aborted, allowing final cleanup or summary actions to run.
        """
        # Guard: return early if the end_schedule is empty
        if not self.schedule.end_schedule:
            return

        # Load the first action in run_schedule
        action = self.schedule.end_schedule[0]

        # Calls action agent method
        call_action(action)

        # Remove the executed action from the run_schedule
        self.schedule.end_schedule.pop(0)