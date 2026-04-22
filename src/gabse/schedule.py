"""
This module contains the simulation scheduling classes.
"""

# %%
# Import required packages
from sortedcontainers import SortedList
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .data import Sensor


# %%
class Action:
    """
    A class representing a scheduled action in the simulation. Each behavior that is to be invoked by an agent is
    scheduled and called using *Action*. It is possible to set an action to be reoccurring by using the *interval*
    parameter.

    Parameters
    ----------
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

    Attributes
    ----------
    tick: float
        The simulation tick at which the action is scheduled to occur.
    agent: Agent | Sensor
        The agent or sensor that will perform the action.
    method: str
        The name of the method to be called on the agent.
    args: list, optional
        The arguments to be passed to the method. Can be None, empty list, or "" if no arguments are needed.
    priority: int, optional
        The priority of the action (lower values indicate higher priority). Default is 0.
    interval: float, optional
        The interval for recurring actions, has to be. If greater than 0, the action will be rescheduled
        after execution. Default is 0.
    """

    def __init__(
        self,
        tick: float,
        agent: "Agent | Sensor",
        method: str,
        args: list = None,
        priority: int = 0,
        interval: float = 0,
    ):
        self.tick = float(tick)
        self.agent = agent
        self.method = method
        self.args = args
        self.priority = int(priority)
        self.interval = float(interval)

        # check so that interval is greater than zero, if not set to zero.
        if self.interval < 0.0:
            self.interval = 0.0

    def __str__(self):
        return f"Action entry:\ntick: {self.tick}, agent: {self.agent}, method: {self.method}, arguments: {self.args}, priority: {self.priority}, interval: {self.interval}"


# %%
class Schedule:
    """
    A class for managing and executing scheduled actions in the simulation. The core of the schedule is a list where
    all planned actions are stored and executed one by one. The schedule uses a SortedList to maintain order of the
    actions based on tick and priority.

    The schedule uses an event-based stepping approach meaning that it does not use fixed tick steps but instead jumps
    between the ticks of the scheduled actions. This means that the tick can step in various lengths depending on the
    actions. A dynamic tick step approach enables greater flexibility and faster simulations.


    Attributes
    ----------
    schedule: SortedList
        A sorted list of scheduled actions, ordered by tick and priority.
    """

    # Creates an empty schedule (list) and tick timer, set to zero
    # List is sorted based on tick value of actions and priority
    def __init__(self):
        self.schedule = SortedList(key=lambda a: (a.tick, a.priority))

    # Schedule method for adding an action in schedule
    def schedule_action(self, action: Action):
        """
        Schedules an action and places it according to its tick and priority

        Parameters
        ----------
        action : Action
            The action object to be scheduled.
        """
        self.schedule.add(action)

    # Method for stepping forward in simulation
    def step(self, old_tick) -> float:
        """
        Steps one entry in the schedule. The step method executes the next action entry and, if reoccurring, re-schedules
        it. It also moves the tick forward one instance, can be the same if multiple actions are scheduled at the same tick.

        Parameters
        ----------
        old_tick : float
            The current tick before stepping.

        Returns
        -------
        tick : float
            The new tick
        """
        # If schedule is empty after removing past actions, return current tick
        if not self.schedule:
            return old_tick

        # Checks if previous actions exist and, if so, removes them
        while self.schedule[0].tick < old_tick:
            self.schedule.pop(0)

        # If schedule is empty after removing past actions, return current tick
        if not self.schedule:
            return old_tick
        # Load the first action in schedule
        action = self.schedule[0]

        # Step to next action tick
        new_tick = action.tick

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
                interval=action.interval,
            )
            self.schedule_action(nextAction)

        # Remove the executed action from the schedule
        self.schedule.pop(0)

        # Return current tick
        return new_tick

    #
    def remove_agent_from_list(self, target):
        """
        Removes all actions related to the target agent. This is useful if an agent has become obsolete, e.g. killed.

        Parameters
        ----------
        target : Agent
            The agent whose actions are to be removed.
        """
        self.schedule = SortedList(
            [action for action in self.schedule if action.agent != target],
            key=lambda action: (action.tick, action.priority),
        )

    def print_schedule(self):
        """
        Prints all actions in the schedule.
        """
        print(f"Printing schedule from tick: {self.tick}:")
        for action in self.schedule:
            print(action)

    def clear_schedule(self):
        """
        Clears the schedule.
        """
        self.schedule.clear()
