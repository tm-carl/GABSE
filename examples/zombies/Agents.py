# %%
# Import of packages
import numpy as np
import src.gabse as gabse


class Person(gabse.Agent):
    def __init__(self, speed, engine, position=np.array([0, 0, 0])):
        self.speed = speed
        self.alive = True

        super().__init__(engine, position)

        freq = 10.0
        self.sensor = gabse.Sensor(engine, self, freq)

        getters = ["position", "alive"]

        a = gabse.Action(
            self.engine.schedule.get_tick() + 1,
            self.sensor,
            "entry",
            getters,
            np.iinfo(np.int32).max,
            self.sensor.get_frequency(),
        )
        self.engine.schedule.schedule_action(a)

    def get_zombies(self):
        zombies = filter(
            lambda x: x.__class__.__name__ == "Zombie", self.engine.context.get_agents()
        )
        return list(zombies)

    def find_closest_zombie(self):
        closestZombie = self.find_neighbours(self.get_zombies(), 1)
        return closestZombie

    # Run method for the Person agent
    def run(self):
        # Find the closest zombie
        ngh = self.find_closest_zombie()

        # Calculate distance vector to the closest zombie
        distVector = ngh.get_position() - self.get_position()

        # Calculate the norm of the distance vector
        norm = np.linalg.norm(distVector)

        # Run away if the zombie is within 10 units and not at the same position
        if norm < 10.0 and norm != 0.0:
            normVector = distVector / np.linalg.norm(distVector)
            runVector = normVector * -1 * self.speed
            self.move_vector(runVector)

        # print(self.getPosition())

    # Getters and Setters
    def get_speed(self):
        return self.speed

    def set_alive(self, boo):
        self.alive = boo

    def get_alive(self):
        return self.alive


class Zombie(gabse.Agent):
    def __init__(self, speed, engine, position=np.array([0, 0, 0])):
        self.speed = speed
        super().__init__(engine, position)

        freq = 10.0
        self.sensor = gabse.Sensor(engine, self, freq)
        getters = ["position"]

        a = gabse.Action(
            engine.schedule.get_tick() + 1,
            self.sensor,
            "entry",
            getters,
            np.iinfo(np.int32).max,
            self.sensor.get_frequency(),
        )
        self.engine.schedule.schedule_action(a)

    def get_persons(self):
        p = filter(
            lambda x: x.__class__.__name__ == "Person", self.engine.context.get_agents()
        )

        return list(filter(lambda x: x.get_alive(), p))

    def find_closest_persons(self, noOfNeighbours: int = 1) -> list | Person:
        closestPerson = self.find_neighbours(self.get_persons(), noOfNeighbours)
        if noOfNeighbours == 1:
            return closestPerson
        else:
            return list(closestPerson)

    def hunt(self):
        ngh = self.find_closest_persons()

        # Check if all people are dead
        if ngh == "":
            self.engine.abort()
        else:
            distVector = ngh.get_position() - self.get_position()

            norm = np.linalg.norm(distVector)

            if norm == 0:
                normVector = np.zeros_like(distVector)
            else:
                normVector = distVector / np.linalg.norm(distVector)

            runVector = normVector * 1 * self.speed
            # print(runVector)

            self.move_vector(runVector)

            if self.get_distance(ngh) < 1.0:
                self.kill(ngh)

            # agents = ["Zombie", "Person"]
            # print(self.engine.context.get_agent_count(agents))

    def kill(self, victim):
        newZombie = Zombie(self.speed, self.engine, victim.get_position())
        self.engine.context.add_agent(newZombie)
        a = gabse.Action(
            self.engine.schedule.get_tick() + 1, newZombie, "hunt", "", 10, 1
        )
        self.engine.schedule.schedule_action(a)

        victim.set_alive(False)
        victim.sensor.entry("position", "alive")
        self.engine.data_logger.store_log(victim)
        sensor = victim.get_sensor()

        self.engine.schedule.remove_agent_from_list(victim)
        self.engine.schedule.remove_agent_from_list(sensor)

        self.engine.context.remove_agent(victim)

        # agents = ["Zombie", "Person"]
        counts = self.get_persons()

        if len(counts) == 0:
            #print("The world is lost... everyone is a zombie")
            self.engine.abort()

    def get_speed(self):
        return self.speed


class Logger(gabse.Agent):
    def __init__(self, engine, position=np.array([0, 0, 0])):
        super().__init__(engine, position)
        sensor = gabse.Sensor(engine, self, 1.0)
        self.set_sensor(sensor)

        a = gabse.Action(
            engine.schedule.get_tick() + 1,
            sensor,
            "entry",
            ["agent_counts"],
            np.iinfo(np.int32).max,
            sensor.get_frequency(),
        )
        self.engine.schedule.schedule_action(a)

    def get_agent_counts(self):
        counts = dict()

        counts["Person"] = self.engine.context.get_agent_count(["Person"])["Person"]
        counts["Zombie"] = self.engine.context.get_agent_count(["Zombie"])["Zombie"]

        return counts

    def get_kpis(self):
        # Calculate the kill rate based on the number of zombies and people in the context at start and current time
        # The initial number of people is stored in the sensor logger when the Logger agent is initialized
        # Gets the initial counts from the sensor logger
        first_key = next(iter(self.sensor.get_logger()))
        init_person_count = self.get_sensor().logger.get(first_key).get("agent_counts")["Person"]
        init_zombie_count = self.get_sensor().logger.get(first_key).get("agent_counts")["Zombie"]

        # Gets the current counts from the context
        last_key = next(reversed(self.sensor.get_logger()))
        curr_person_count = self.get_sensor().logger.get(last_key).get("agent_counts")["Person"]
        curr_zombie_count = self.get_sensor().logger.get(last_key).get("agent_counts")["Zombie"]

        # Calculate the kill rate as the change in zombie count divided by the change in person count, multiplied by 100 to get a percentage
        if init_person_count - curr_person_count == 0:
            kill_ratio = 0.0
        else:
            kill_ratio = ((curr_zombie_count - init_zombie_count) / (init_person_count - curr_person_count)) * 100


        return {
            "person_count": self.engine.context.get_agent_count(["Person"])["Person"],
            "zombie_count": self.engine.context.get_agent_count(["Zombie"])["Zombie"],
            "kill_ratio": kill_ratio,
            "kill_rate": round((curr_zombie_count - init_zombie_count) / self.engine.schedule.get_tick(), 4)
        }