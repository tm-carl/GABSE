from gabse import context
import numpy as np
import gabse

class Person(gabse.Agent):
    def __init__(self, speed, engine, position=np.array([0, 0, 0])):
        self.speed = speed
        self.alive = True

        super().__init__(engine, position)

        freq = 10.0
        sensor = gabse.Sensor(engine, self, freq)
        self.add_sensor(sensor)

        getters = ["position", "alive"]

        a = gabse.Action(self.engine.get_tick() + 1, sensor, "entry", getters, np.iinfo(np.int32).max, sensor.get_frequency())
        self.engine.schedule.schedule_action(a)

    def get_zombies(self):
        zombies = filter(lambda x: x.__class__.__name__ == "Zombie", self.engine.context.get_agents())
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
        sensor = gabse.Sensor(engine, self, freq)

        self.add_sensor(sensor)
        getters = ["position"]

        a = gabse.Action(engine.get_tick() + 1, sensor, "entry", getters, np.iinfo(np.int32).max, sensor.get_frequency())
        self.engine.schedule.schedule_action(a)

    def get_persons(self):
        p = filter(lambda x: x.__class__.__name__ == "Person", self.engine.context.get_agents())

        return list(filter(lambda x: x.get_alive(), p))

    def find_closest_person(self):
        closestPerson = self.find_neighbours(self.get_persons(), 1)
        return closestPerson

    def hunt(self):
        ngh = self.find_closest_person()

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

            if self.get_distance(self, ngh) < 1.0:
                self.kill(ngh)

            # agents = ["Zombie", "Person"]
            # print(self.engine.context.get_agent_count(agents))

    def kill(self, victim):
        newZombie = Zombie(self.speed, self.engine, victim.get_position())
        self.engine.context.add_agent(newZombie)
        a = gabse.Action(self.engine.get_tick() + 1, newZombie, "hunt", "", 10, 1)
        self.engine.schedule.schedule_action(a)

        victim.set_alive(False)
        sensor = victim.get_sensor()
        self.engine.schedule.remove_agent_from_list(victim)
        self.engine.schedule.remove_agent_from_list(sensor)

        agents = ["Zombie", "Person"]
        counts = self.get_persons()

        if len(counts) == 0:
            print("The world is lost... everyone is a zombie")
            self.engine.abort()

    def get_speed(self):
        return self.speed

class Logger(gabse.Agent):
    def __init__(self, engine, position=np.array([0, 0, 0])):
        super().__init__(engine, position)
        sensor = gabse.Sensor(engine, self, 1.0)
        self.add_sensor(sensor)

        a = gabse.Action(engine.get_tick() + 1, sensor, "entry", ["agent_counts"], np.iinfo(np.int32).max, sensor.get_frequency())
        self.engine.schedule.schedule_action(a)

    def get_agent_counts(self):
        return self.engine.context.get_agent_count()