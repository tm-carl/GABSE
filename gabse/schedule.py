"""
@author: Carl Toller Melén

"""

# %%
# Import required packages
from sortedcontainers import SortedList

# %%
# Action class for representing scheduled actions

class Action:
    """
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
    """
    def __init__(self, tick, agent, method, args=None, priority=0, interval=0):
        self.tick = float(tick)
        self.agent = agent
        self.method = method
        self.args = args
        self.priority = int(priority)
        self.interval = float(interval)

    def __str__(self):
        return f"Action entry:\ntick: {self.tick}, agent: {self.agent}, method: {self.method}, arguments: {self.args}, priority: {self.priority}, interval: {self.interval}"


# %%
# Schedule class for managing and executing scheduled actions
class Schedule:

    # Creates an empty schedule (list) and tick timer, set to zero
    # List is sorted based on tick value of actions and priority
    def __init__(self, tick):
        self.schedule = SortedList(key=lambda a: (a.tick, a.priority))
        self.tick = tick

    # Schedule method for adding an action in schedule
    def schedule_action(self, action: Action):
        self.schedule.add(action)

    # Method for stepping forward in simulation
    def step(self):
        # If schedule is empty, return current tick
        if not self.schedule:
            return self.tick

        # Checks if previous actions exist and, if so, removes them
        while self.schedule[0].tick < self.tick:
            self.schedule.pop(0)

        # If schedule is empty after removing past actions, return current tick
        if not self.schedule:
            return self.tick

        # Load the first action in schedule
        action = self.schedule[0]

        # Step to next action tick
        self.tick = action.tick

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
                interval=action.interval
            )
            self.schedule_action(nextAction)

        # Remove the executed action from the schedule
        self.schedule.pop(0)

        #Return current tick
        return self.tick


    # Filter out all actions related to the target agent
    def remove_agent_from_list(self, target):
        self.schedule = SortedList(
            [action for action in self.schedule if action.agent != target],
            key=lambda action: (action.tick, action.priority)
        )

    def get_schedule(self):
        return self.schedule

    def print_schedule(self):
        for action in self.schedule:
            print(action)

    def get_tick(self):
        return self.tick

    def get_size(self):
        return len(self.schedule)

    def clear_schedule(self):
        self.schedule.clear()
