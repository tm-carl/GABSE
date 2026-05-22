"""
This module contains the simulation scheduling classes.
"""

# %%
# Import required packages
from sortedcontainers import SortedList

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
        The arguments to be passed to the method. Can be None if no arguments are needed.
    priority: int, optional
        The priority of the action (lower values indicate higher priority). Default is 0.
    interval: float, optional
        The interval for recurring actions, has to be. If greater than 0, the action will be rescheduled
        after execution. Default is 0.
    """

    def __init__(
        self,
        tick: float,
        agent,
        method: str,
        args: list | None = None,
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
    run_schedule: SortedList
        A sorted list of scheduled actions, ordered by tick and priority.

    post_process: SortedList
        A sorted list of end actions that are executed at the end of the simulation, ordered by priority.
    """

    def __init__(self):
        self.run_schedule = SortedList(key=lambda a: (a.tick, a.priority))
        self.post_process = SortedList(key=lambda a: a.priority)

    # Schedule method for adding an action in run_schedule
    def schedule_action(self, action: Action):
        """
        Schedules an action and places it according to its tick and priority.
        Raises ``AttributeError`` immediately if the method name does not exist on
        the agent, so typos are caught at scheduling time rather than execution time.

        Parameters
        ----------
        action : Action
            The action object to be scheduled.

        Raises
        ------
        AttributeError
            If the agent does not have a method matching *action.method*.
        """
        if not hasattr(action.agent, action.method):
            raise AttributeError(
                f"Agent '{type(action.agent).__name__}' has no method '{action.method}'."
            )
        self.run_schedule.add(action)

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
            If the agent does not have a method matching *action.method*.
        """
        if not hasattr(action.agent, action.method):
            raise AttributeError(
                f"Agent '{type(action.agent).__name__}' has no method '{action.method}'."
            )
        self.post_process.add(action)

    def remove_agent_from_list(self, target, remove_post_process: bool = True):
        """
        Removes all actions related to the target agent. This is useful if an agent has become obsolete, e.g. killed.

        Parameters
        ----------
        target : Agent
            The agent whose actions are to be removed.

        remove_post_process : bool, optional
            Whether to also remove the agent's end actions. Default is True.
        """
        self.run_schedule = SortedList(
            [action for action in self.run_schedule if action.agent != target],
            key=lambda action: (action.tick, action.priority),
        )

        if remove_post_process:
            self.post_process = SortedList(
                [action for action in self.post_process if action.agent != target],
                key=lambda action: (action.tick, action.priority),
            )

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