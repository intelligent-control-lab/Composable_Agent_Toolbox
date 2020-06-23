import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
 
    def __init__(self, name, init_x):
        self.name = name
        self._x = init_x

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

    