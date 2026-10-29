"""
This module contains the simulation engine class.
"""
import reprlib

from tqdm import tqdm

from .agent import Sensor
from .data import DataCollector
from .context import Context
from .schedule import Schedule, Action

#%%

def call_action(action: Action):
    """
    Calls the method specified in the action on the agent with the provided arguments. For more details about the Action class, see the documentation for the Action class.

    Parameters
    ----------
    action : Action
        The action to be called.
    """
    # Get the method from the agent
    method = getattr(action.agent, action.method)

    # Get the arguments for the method, if any
    args = []

    if action.args is not None:
        args = list(action.args)

    # Check if the method is callable and call it
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
    context : Context, optional
        The context to be used, if custom. Default is to use the built-in context.
    progress_bar : bool, optional
        Whether to display a progress bar during the simulation run. Default is False.

    Attributes
    ----------
    tick : float
        The current simulation tick.
    model_time : float
        The total time for which the simulation will run.
    context : Context | Any
        The context containing the agents and environment of the simulation. Can also be a child class of Context class.
    schedule : Schedule
        The registry of actions to be executed, but during the simulation and post-simulation.
    aborted : bool
        Whether the simulation is aborted or not.
    data_logger : DataCollector
        The data collector for the simulation, which collects KPIs and log data after the simulation run.
    """

    def __init__(
            self,
            model_time: float,
            context: Context,
            progress_bar: bool = False,
            ):

        self.tick = 0.0
        self.model_time = model_time
        self.schedule = Schedule()
        self.data_logger = DataCollector()
        self.aborted = False
        self.context = context
        self.progress_bar = progress_bar

        if progress_bar:
            self.last_time = 0.0
            self.pbar = tqdm(total=model_time, desc="Simulation progress bar", leave=True, position=0)

    # ----------------------
    # Run control
    # ----------------------

    def run(self, no_arg_out:int = 0) -> None | dict | tuple:
        """
        Runs the simulation until reached model time, schedule is empty, or simulation is aborted internally.

        Parameters
        ----------
        no_arg_out : int, optional
            The number of output arguments to return. Default is 0 (returns nothing). If 1, returns the collected KPIs
            as a dictionary. If 2, returns a tuple containing the collected KPIs and the collected log data as dictionaries.

        Returns
        -------
        None | dict | tuple
            The return value depends on the value of *no_arg_out*. If *no_arg_out* is 0, the method returns `None`.
            If *no_arg_out* is 1, it returns a dictionary containing the collected KPIs. If *no_arg_out* is 2, it
            returns a tuple containing the collected KPIs and the collected log data as dictionaries.
        None
            If *no_arg_out* is 0, the method returns nothing.
        KPIs : dict
            The collected KPIs from the simulation, returned if *no_arg_out* is 1 or 2.
        data : dict
            The collected log data from the simulation, returned if *no_arg_out* is 2.
        """

        # Continuously steps through the run_schedule until it is empty or the simulation is aborted. step() drops the
        # actions scheduled after the model time, so the tick never exceeds it.
        while self.schedule.run_schedule and not self.aborted:
            self.step()

            # Update progress bar if enabled
            if self.progress_bar:
                time_increment = self.tick - self.last_time

                if time_increment > 0:
                    self.pbar.update(time_increment)
                    self.last_time = self.tick

        # At the end of the simulation, iterate through end actions assigned (allowing for final events to occur)
        while len(self.schedule.post_process) > 0:
            self.end_step()

        # Return nothing, the KPIs, or the KPIs and the log data, based on the value of no_arg_out
        if no_arg_out == 0: # default, returns nothing
            return None

        if no_arg_out not in (1, 2): # returns nothing and prints a warning message.
            print("Warning: no_arg_out should be 0, 1, or 2. Returning nothing.")
            return None

        self.data_logger.collect_kpis(tick=self.tick, context=self.context, agents=self.context.agents)

        if no_arg_out == 1: # returns collected KPIs as a dictionary
            return self.data_logger.export_kpis()

        # no_arg_out == 2, activated as last option if none of previous have been true.
        # returns collected KPIs and log data as a tuple of dictionaries. Logs of retired agents
        # (removed with keep_log=True) are included. A live agent wins if an id is shared.
        retired_agents = getattr(self.context, "retired_agents", {})
        self.data_logger.collect_data(agents={**retired_agents, **self.context.agents})
        return self.data_logger.export_kpis(), self.data_logger.export_data()

    def abort(self):
        """
        Aborts the simulation. This method clears the run_schedule, sets the aborted flag to True, and updates the progress bar to indicate that the simulation has been aborted effectively making the *run()* method exit its main loop and stop the simulation, including post-processing. It is safe to call from inside an action: the action being executed finishes normally and is not rescheduled.
        """
        # Cancelling a queued action removes it from the list, so the actions are collected first
        for action in list(self.schedule.run_schedule):
            self.schedule.cancel_action(action)

        self.aborted = True

        # Updates the progress bar to read 100%
        if self.progress_bar:
            self.pbar.total = self.last_time
            self.pbar.set_description("Simulation progress bar (Aborted)")
            self.pbar.refresh()
            self.pbar.close()

    # ----------------------
    # Action stepping
    # ----------------------

    def step(self):
        """
        Steps one entry in the run_schedule. The step method executes the next action entry and, if reoccurring, re-schedules it. It also moves the tick forward one instance, can be the same if multiple actions are scheduled at the same tick.

        The action is taken out of the run_schedule before it is executed, so the action may freely schedule or cancel other actions (including its own agent's actions) and may call *abort()*. A recurring action is re-queued as the same object with its tick advanced by its interval, which keeps its priority.
        """
        schedule = self.schedule
        queue = schedule.run_schedule

        # Guard: Drop cancelled actions, stale actions whose tick is earlier than the current engine tick, and actions
        # scheduled after the specified model time (i.e., beyond the simulation's time horizon)
        while queue and (queue[0].cancelled or queue[0].tick < self.tick or queue[0].tick > self.model_time):
            schedule.cancel_action(queue[0])

        # Guard: return early if no valid action remains
        if not queue:
            return

        # Take the first action (sorted by tick and then by priority) out of the run_schedule
        action = schedule.take_next(queue)

        # Step to the action tick in the simulation
        self.tick = action.tick

        # Call the action agent method. A failing action gets a note telling where it happened, then the exception is re-raised unchanged
        try:
            call_action(action)
        except BaseException as exc:
            if isinstance(exc, Exception):
                exc.add_note(self._error_note(action, "run"))
            schedule.finish_action(action, False)
            raise

        # Reschedule the action if recurring, unless it was cancelled or the simulation was aborted during the call
        schedule.finish_action(action, action.interval > 0.0 and not action.cancelled and not self.aborted)

    def end_step(self):
        """
        Executes the next action in the end-of-simulation queue (*post_process*). Called by the engine after the main *run_schedule* is exhausted or the simulation is aborted, allowing final cleanup or summary actions to run.
        """
        schedule = self.schedule

        # Guard: return early if the post_process is empty
        if not schedule.post_process:
            return

        # Take the first action out of the post_process before executing it
        action = schedule.take_next(schedule.post_process)

        # Call the action agent method, unless it was cancelled
        try:
            if not action.cancelled:
                call_action(action)
        except Exception as exc:
            exc.add_note(self._error_note(action, "post-process"))
            raise
        finally:
            schedule.finish_action(action, False)

    # ----------------------
    # Internal helpers
    # ----------------------
    
    def _error_note(self, action: Action, phase: str) -> str:
        """Builds the note that tells which action was executing when an exception was raised."""
        agent = action.agent
        owner = agent.parent if isinstance(agent, Sensor) else agent
        owner_text = f"{type(owner).__name__} {getattr(owner, 'agent_id', '?')}"
        target_text = f"Sensor of {owner_text}" if isinstance(agent, Sensor) else owner_text

        # Arguments can be large objects, so they are shortened, and a failing repr must not hide the real exception
        try:
            args_text = reprlib.repr(list(action.args)) if action.args else "[]"
        except Exception:
            args_text = "<unrepresentable>"

        return (
            f"gabse: action failed in {phase} at tick {self.tick} -> {target_text}.{action.method}(args={args_text}), "
            f"priority {action.priority}, interval {action.interval}"
        )