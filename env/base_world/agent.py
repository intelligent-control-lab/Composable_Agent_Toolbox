import numpy as np
import math
from abc import ABC, abstractmethod

class Agent(ABC):
 
    def __init__(self, name, spec, collision=True):
        self.name = name
        self._x = np.array(spec['init_x']).reshape(-1, 1)
        self.xdim = self._x.shape[0]
        self.collision = collision
        self.broadcast = {}
        self.has_heading = False

    @abstractmethod
    def forward(self):
        pass
    
    @property
    def state(self):
        return self._x
    
    @abstractmethod
    def pos(self):
        pass

    @abstractmethod
    def vel(self):
        pass
    
    @property
    def info(self):
        info = {"state": self.state, "pos":self.pos, "vel":self.vel}
        return info
    
class BB8Agent(Agent):
    
    def _f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def _g(self, x):
        B = np.zeros((4,2))
        B[2,0] = 0.5
        B[3,1] = 0.5
        return B

    def forward(self, action, dt):
        # x = [x y dx dy], u = [ax ay]
        u = action['control']
        dot_x   = self._f(self._x) + (self._g(self._x) @ np.vstack(u))
        self._x = self._x + (dot_x * dt)
        
        self.broadcast = action["broadcast"] if "broadcast" in action.keys() else {}

    @property
    def pos(self):
        return self._x[[0,1]]
    
    @property
    def vel(self):
        return self._x[[2,3]]

class UnicycleAgent(Agent):
    '''
        state _x is [x, y, t]
        action is [v, w], v -> velocity, w -> heading velocity
        [xdot, ydot, tdot] = [vcost, vsint, w]
    '''

    def __init__(self, name, spec, collision=True):
        super().__init__(name, spec, collision=collision)

        self.has_heading = True

    def forward(self, action, dt):
        
        u = action['control'] # v, w
        v, w = u.reshape(-1)

        t = self._x[2, 0]
        xdot = np.array([v*math.cos(t), v*math.sin(t), w]).reshape(-1, 1)

        self._x[:self.xdim//2] += xdot*dt
        self._x[2] = math.atan2(math.sin(self._x[2]), math.cos(self._x[2]))
        self._x[-self.xdim//2:] = xdot

        self.broadcast = action["broadcast"] if "broadcast" in action.keys() else {}
    
    @property
    def pos(self):
        return self._x[:self.xdim//2]

    @property
    def vel(self):
        return self._x[-self.xdim//2:]

class GoalAgent(BB8Agent):
    """The goal agent.
    This agent is a virtual agent represents the goal of a real agent.
    This agent only flash to a new place when the real agent reaches it.
    The reason we inheritate it from BB8Agent is to make it possible to be a 
    dynamic goal in the future.
    """
    def __init__(self, name, hunter, goal_list, reaching_eps, collision=False):
        self.name = name
        self._x = np.zeros((4,1))
        self.goal_list = goal_list
        self.goal_idx = 0
        self.hunter = hunter
        self.reaching_eps = reaching_eps
        self.collision = collision
        self._set_pos()
        self.broadcast = {}

    def _set_pos(self):
        self._x[[0,1]] = np.vstack(self.goal_list[self.goal_idx])

    def forward(self):
        if np.max(abs(self.pos - self.hunter.pos)) < self.reaching_eps:
            self.goal_idx = min(len(self.goal_list)-1, self.goal_idx+1)
            self._set_pos()

    @property
    def info(self):
        info = {"state": self.state, "pos":self.pos, "vel":self.vel, "count":self.goal_idx}
        return info
