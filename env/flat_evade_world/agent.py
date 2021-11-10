import numpy as np
from abc import ABC, abstractmethod
from env.base_world.agent import Agent

    
class BB8Agent(Agent):
    
    def _f(self, x):
        return np.vstack([x[2], x[3], 0, 0])

    def _g(self, x):
        B = np.matrix(np.zeros((4,2)))
        B[2,0] = 0.5
        B[3,1] = 0.5
        return B

    def forward(self, action, dt):
        # x = [x y dx dy], u = [ax ay]
        u = action['control']
        dot_x   = self._f(self._x) + (self._g(self._x)*np.vstack(u))
        self._x = self._x + (dot_x * dt)
        
        self.broadcast = action["broadcast"] if "broadcast" in action.keys() else {}


    @property
    def pos(self):
        return self._x[[0,1]]
    
    @property
    def vel(self):
        return self._x[[2,3]]


# todo change to adversarial agent
class GoalAgent(BB8Agent):
    """The goal agent.
    This agent is a virtual agent represents the goal of a real agent.
    This agent only flash to a new place when the real agent reaches it.
    The reason we inheritate it from BB8Agent is to make it possible to be a 
    dynamic goal in the future.

    The goal is either specified in absolute location, or relative to
    hunter and adversarial target (the goal will extend from hunter to adv)

    """
    def __init__(self, name, hunter, adversarial_target,
                    reaching_eps, collision=False):
        self.name = name
        self._x = np.zeros((4,1))
        self.adversarial_target = adversarial_target
        self.hunter = hunter
        self.reaching_eps = reaching_eps
        self.collision = collision
        self.broadcast = {}
        self.goal_reached = 0

        # init pos
        self._set_pos()

    def _set_pos(self):
        if self.adversarial_target is None:
            self._x[[0,1]] = np.vstack(self.hunter.pos) + np.random.random()*10
        else:
            adv_pos = self.adversarial_target.pos
            hunt_pos = self.hunter.pos
            n = np.linalg.norm(adv_pos - hunt_pos)
            if n == 0:
                u = adv_pos - hunt_pos
            else:
                u = (adv_pos - hunt_pos)/n
            self._x[[0,1]] = np.clip(np.vstack(adv_pos + 10*u) + np.random.random()*10, 0, 100)

    def forward(self):
        if np.max(abs(self.pos - self.hunter.pos)) < self.reaching_eps:
            self.goal_reached += 1
            self._set_pos()

    @property
    def info(self):
        info = {
            "state": self.state,
            "pos":self.pos,
            "vel":self.vel,
            "count": self.goal_reached,
            "adv_target":self.adversarial_target.name if self.adversarial_target is not None else "none"
            }
        return info
