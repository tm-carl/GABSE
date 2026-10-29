"""
This module contains the simulation scheduling classes.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

from sortedcontainers import SortedList

from .agent import Sensor

if TYPE_CHECKING:
    from .agent import Agent


# %%
class Action:
    """
    A class representing a scheduled action in the simulation. Each behavior that is to be executed by an agent is scheduled and called using this *Action* class. It is possible to set an action to be reoccurring by using the *interval* parameter.

    Parameters
    ----------
    tick: float
        The simulation tick at which the action is scheduled to occur. Must be a finite number.
    agent: Agent | Sensor
        The agent or sensor that will perform the action.
    method: str
        The name of the method to be called on the agent.
    args: list, optional
        The arguments to be passed to the method. Can be None if no arguments are needed.
    priority: int, optional
        The priority of the action (lower values indicate higher priority). Default is 0.
    interval: float, optional
        The interval for recurring actions, has to be a non-negative number. If greater than 0, the action will be rescheduled after execution. Default is 0.

    Raises
    ------
    ValueError
        The tick is NaN or infinite.

    Attributes
    ----------
    tick: float
        The simulation tick at which the action is scheduled to occur.
    agent: Agent | Sensor
        The agent or sensor that will perform the action.
    method: str
        The name of the method to be called on the agent.
    args: list, optional
        The arguments to be passed to the method. Can be None if no arguments are needed.
    priority: int, optional
        The priority of the action (lower values indicate higher priority). Default is 0.
    interval: float, optional
        The interval for recurring actions, has to be a non-negative number. If greater than 0, the action will be rescheduled after execution. Default is 0.
    cancelled: bool
        Whether the action has been cancelled. A cancelled action is never executed and, if recurring, never rescheduled. Setting the flag to True is enough: the engine skips the action when it reaches it. Use *Schedule.cancel_action()* or *Schedule.cancel_agent_actions()* to cancel and also remove it from the schedule right away.

    Notes
    -----
    A recurring action is executed as the same *Action* object each time; the engine only advances its *tick*. Hence its *priority* is kept, and the object can be used as a handle to cancel the whole series.
    """

    def __init__(
        self,
        tick: float,
        agent: "Agent" | "Sensor",
        method: str,
        args: list | None = None,
        priority: int = 0,
        interval: float = 0,
    ):
        self.tick = float(tick)

        # Guard: a NaN tick would silently end the simulation, and an infinite tick would never run
        if not math.isfinite(self.tick):
            raise ValueError(f"Action tick must be finite, got {self.tick} for {type(agent).__name__}.{method}().")

        self.agent = agent
        self.method = method
        self.args = args
        self.priority = int(priority)
        self.interval = float(interval)

        # Guard: check so that interval is greater than zero, if not set to zero.
        if self.interval < 0.0:
            self.interval = 0.0

        self.cancelled = False

        # The SortedList currently holding this action; None while it executes or after it finished. Managed by Schedule.
        self._queued_in = None

    # Override the string representation of the Action class for better readability
    def __str__(self):
        return f"Action entry:\ntick: {self.tick}, agent: {self.agent}, method: {self.method}, arguments: {self.args}, priority: {self.priority}, interval: {self.interval}"



# %%
class Schedule:
    """
    A class for managing and executing scheduled actions in the simulation. The core of the schedule is a list where all planned actions are stored and executed one by one. The schedule uses a SortedList to maintain order of the actions based on tick and priority. The schedule uses an event-based stepping approach meaning that it does not use fixed tick steps but instead jumps between the ticks of the scheduled actions. This means that the tick can step in various lengths depending on the actions. A dynamic tick step approach enables greater flexibility and faster simulations.

    The schedule also has a separate list for end actions that are executed at the end of the simulation, after all regular actions have been executed. This allows for cleanup or finalization tasks to be performed after the main simulation loop has completed.


    Attributes
    ----------
    run_schedule: SortedList
        A sorted list of scheduled actions, ordered by tick and priority.

    post_process: SortedList
        A sorted list of end actions that are executed at the end of the simulation, ordered by priority.

    Notes
    -----
    The schedule keeps an index of every queued or currently executing action per owning agent (a *Sensor* counts as part of its parent agent). This makes *cancel_agent_actions()* cheap and safe to call at any time, also from inside an action that is being executed, because the lists are never rebuilt and the executing action is not part of *run_schedule* while it runs.

    The methods are grouped as: scheduling and cancelling, the interface used by the *Engine*, and printing.
    """

    def __init__(self):
        self.run_schedule = SortedList(key=lambda action: (action.tick, action.priority))
        self.post_process = SortedList(key=lambda action: action.priority)

        # Actions that are queued or currently executing, grouped by id() of their owning agent
        self._pending_by_owner: dict[int, set[Action]] = {}

    # ----------------------
    # Scheduling and cancelling
    # ----------------------

    def schedule_action(self, action: Action):
        """
        Schedules an action and places it in the *run_schedule* according to its tick and priority.

        Parameters
        ----------
        action : Action
            The action object to be scheduled.

        Raises
        ------
        AttributeError
            The agent does not have a method matching *action.method*.
        ValueError
            The action is already scheduled.
        """

        if not hasattr(action.agent, action.method):
            raise AttributeError(f"Agent '{type(action.agent).__name__}' has no method '{action.method}'.")

        if action._queued_in is not None:
            raise ValueError("The action is already scheduled; create a new Action to schedule it again.")

        # A Sensor counts as part of its parent agent
        owner = action.agent.parent if isinstance(action.agent, Sensor) else action.agent

        self.run_schedule.add(action)
        action._queued_in = self.run_schedule
        self._pending_by_owner.setdefault(id(owner), set()).add(action)

    def schedule_post_process(self, action: Action):
        """
        Schedules an end action that will be executed at the end of the simulation, after all regular actions have been executed.

        Parameters
        ----------
        action : Action
            The action object to be scheduled as an end action.

        Raises
        ------
        AttributeError
            The agent does not have a method matching *action.method*.
        ValueError
            The action is already scheduled.
        """

        if not hasattr(action.agent, action.method):
            raise AttributeError(f"Agent '{type(action.agent).__name__}' has no method '{action.method}'.")

        if action._queued_in is not None:
            raise ValueError("The action is already scheduled; create a new Action to schedule it again.")

        # A Sensor counts as part of its parent agent
        owner = action.agent.parent if isinstance(action.agent, Sensor) else action.agent

        self.post_process.add(action)
        action._queued_in = self.post_process
        self._pending_by_owner.setdefault(id(owner), set()).add(action)

    def cancel_action(self, action: Action) -> bool:
        """
        Cancels a single action. A queued action is removed from its list at once. If the action is currently being executed it is flagged, so the engine will not reschedule it if it is recurring. For a recurring action this ends the whole series.

        Parameters
        ----------
        action : Action
            The action to cancel.

        Returns
        -------
        was_pending : bool
            True if the action was still pending (queued or executing), False if it had already finished.
        """
        owner = action.agent.parent if isinstance(action.agent, Sensor) else action.agent
        pending = self._pending_by_owner.get(id(owner), ())
        was_pending = action in pending

        action.cancelled = True

        # A queued action is removed from its list and from the index. An executing action stays in the index until the engine finishes it
        if action._queued_in is not None:
            action._queued_in.remove(action)
            action._queued_in = None
            pending.discard(action)

            if not pending:
                del self._pending_by_owner[id(owner)]

        return was_pending

    def cancel_agent_actions(self, target, method: str | None = None, include_post_process: bool = True) -> int:
        """
        Cancels the pending actions of an agent. Actions of the agent's *Sensor* are included. It is safe to call this from inside an action, also for the agent's own actions, and it never disturbs the actions of other agents. Use it when an agent becomes obsolete (e.g. leaves the simulation) or when a planned behavior is replaced.

        Parameters
        ----------
        target : Agent | Sensor
            The agent whose actions are to be cancelled. If a *Sensor* is given, only the actions of that sensor are cancelled.
        method : str, optional
            If given, only actions calling this method name are cancelled. Default is None (all actions).
        include_post_process : bool, optional
            Whether to also cancel the end actions in *post_process*. Default is True.

        Returns
        -------
        cancelled_count : int
            The number of actions that were cancelled.
        """
        owner = target.parent if isinstance(target, Sensor) else target
        pending_actions = list(self._pending_by_owner.get(id(owner), ()))
        sensor_only = isinstance(target, Sensor)
        cancelled_count = 0

        for action in pending_actions:
            if sensor_only and action.agent is not target:
                continue
            if method is not None and action.method != method:
                continue
            if not include_post_process and action._queued_in is self.post_process:
                continue

            if self.cancel_action(action):
                cancelled_count += 1

        return cancelled_count

    # ----------------------
    # Interface used by the Engine
    # ----------------------

    def take_next(self, queue: SortedList) -> Action:
        """
        Takes the first action out of *queue* so that it can be executed. The action stays registered as pending until *finish_action()* is called, so it can still be cancelled while it runs.

        Parameters
        ----------
        queue : SortedList
            The list to take from, *run_schedule* or *post_process*.

        Returns
        -------
        action : Action
            The first action of the list.
        """
        action = queue.pop(0)
        action._queued_in = None

        return action

    def finish_action(self, action: Action, reschedule: bool):
        """
        Ends the execution of an action taken with *take_next()*. If *reschedule* is True the action is put back in *run_schedule* with its tick advanced by its interval, which keeps its priority and identity. Otherwise it is no longer pending.

        Parameters
        ----------
        action : Action
            The action that was executed.
        reschedule : bool
            Whether to schedule the next occurrence of a recurring action.
        """
        # A re-queued action is still registered in the index, so only the list needs updating
        if reschedule:
            action.tick += action.interval
            self.run_schedule.add(action)
            action._queued_in = self.run_schedule
            return

        owner = action.agent.parent if isinstance(action.agent, Sensor) else action.agent
        pending = self._pending_by_owner.get(id(owner))

        if pending is not None:
            pending.discard(action)

            if not pending:
                del self._pending_by_owner[id(owner)]

    # ----------------------
    # Printing
    # ----------------------

    def print_run_schedule(self):
        """
        Prints all actions in the run_schedule.
        """
        for action in self.run_schedule:
            print(action)

    def print_end_schedule(self):
        """
        Prints all actions in the post_process.
        """
        for action in self.post_process:
            print(action)
