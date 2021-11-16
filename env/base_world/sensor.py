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

# todo upgrade to anchor+geometry for both agent/obstacle
# todo return a list of polytopes
class RadarSensor(Sensor):
    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        ret = {}
        for name, agent in self.all_agents.items():
            # todo change collision name
            if agent != self.agent and agent.collision:
                d = min(len(agent.pos), len(self.agent.pos))
                ret[name] = {
                    "rel_pos": self._gaussian_noise(
                        agent.pos[:d] - self.agent.pos[:d], self.noise_var),
                    "rel_vel": self._gaussian_noise(
                        agent.vel[:d] - self.agent.vel[:d], self.noise_var)
                }
        
        return ret


class GoalSensor(Sensor):

    def __init__(self, agent, all_agents, spec):
        super().__init__(agent, all_agents, spec)
        self.noise_var = spec['noise_var']

    def measure(self):
        goal_dim = len(self.all_agents[self.agent.name+"_goal"].pos)
        ret = {
            "rel_pos": self._gaussian_noise(
                self.all_agents[self.agent.name+"_goal"].pos - self.agent.pos[:goal_dim],
                self.noise_var),
            "rel_vel": self._gaussian_noise(
                self.all_agents[self.agent.name+"_goal"].vel - self.agent.vel[:goal_dim],
                self.noise_var)
        }
        return ret
