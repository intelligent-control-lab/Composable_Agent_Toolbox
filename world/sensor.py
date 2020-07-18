import numpy as np
from abc import ABC, abstractmethod

class Sensor(ABC):
 
    def __init__(self, agent, all_agents, spec):
        self.agent = agent
        self.all_agents = all_agents
        self.alias = spec["alias"]

    def _gaussian_noise(self, x, variance):
        return x + (np.random.random_sample(x.shape) * variance - variance/2)

    @abstractmethod
    def measure(self):
        pass

class RadioSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
    
    def measure(self):
        broadcast = {}
        for name, agent in self.all_agents.items():
            if agent != self.agent and len(agent.broadcast) > 0:
                broadcast[name] = agent.broadcast
        return broadcast


class PVSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        ret = {
            "pos": self._gaussian_noise(self.agent.pos, self.noise_var),
            "vel": self._gaussian_noise(self.agent.vel, self.noise_var)
        }
        return ret

class StateSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        ret = {
            "state": self._gaussian_noise(self.agent.state, self.noise_var),
        }
        return ret


class RadarSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        distances = {}
        for name, agent in self.all_agents.items():
            if agent != self.agent and agent.collision:
                distances[name] = self._gaussian_noise(agent.pos - self.agent.pos, self.noise_var)
        return distances


class GoalSensor(Sensor):

    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        ret = {
            "rel_pos": self._gaussian_noise(self.all_agents[self.agent.name+"_goal"].pos - self.agent.pos, self.noise_var),
            "rel_vel": self._gaussian_noise(self.all_agents[self.agent.name+"_goal"].vel - self.agent.vel, self.noise_var)
        }
        return ret
