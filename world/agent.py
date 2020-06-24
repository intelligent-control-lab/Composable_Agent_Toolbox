import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
 
    def __init__(self, name, init_x, collision=True):
        self.name = name
        self._x = init_x
        self.collision = collision

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

class BB8Agent(Agent):
    
    def _f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def _g(self, dt):
        B = np.matrix(np.zeros((4,2)))
        B[0,0] = dt/2
        B[1,1] = dt/2
        B[2,0] = 1
        B[3,1] = 1
        return B

    def forward(self, u, dt):
        # x = [x y dx dy], u = [ax ay]
        dot_x = self._f(self._x) + self._g(dt) * np.vstack(u)
        self._x = self._x + dot_x * dt
    
    @property
    def pos(self):
        return self._x[[0,1]]
    
    @property
    def vel(self):
        return self._x[[2,3]]



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

    def _set_pos(self):
        self._x[[0,1]] = np.vstack(self.goal_list[self.goal_idx])

    def forward(self):
        if np.max(abs(self.pos - self.hunter.pos)) < self.reaching_eps:
            self.goal_idx = min(len(self.goal_list)-1, self.goal_idx+1)
            self._set_pos()
        