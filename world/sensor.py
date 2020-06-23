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

class RadarSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        distances = {}
        for agent in self.all_agents:
            if agent != self.agent:
                distances[agent.name] = self._gaussian_noise(agent.pos - self.agent.pos, self.noise_var)
        return distances
