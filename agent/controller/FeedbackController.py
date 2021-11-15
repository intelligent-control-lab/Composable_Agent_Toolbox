import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from utils import GoalType

class FeedbackController(ABC):
    '''
        Feedback Controller Base
    '''

    def __init__(self,
        model):

        self.model = model

    @abstractmethod
    def _control(self, processed_data: dict, error: np.ndarray) -> np.ndarray:
        '''
            Can be model inverse or other control algo
        '''
        pass
    
    def __call__(self,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType,
        state_dimension: int
    ) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''

        # goal -> state error (pos + vel)
        e = self.model.compute_error(
            processed_data=processed_data, goal=goal,
            goal_type=goal_type, state_dimension=state_dimension)
        
        # state error -> action
        u = self._control(processed_data=processed_data, error=e)

        return u


class NaiveFeedbackController(FeedbackController):

    def __init__(self, spec, model):
        super().__init__(model)

        # weights
        self.kp = spec["kp"]
        self.kv = spec["kv"]
        self.u_max = spec["u_max"]

    def _control(self, processed_data: dict, error: np.ndarray) -> np.ndarray:
        '''
            P control on both pos and vel
            Then use control model to convert to action
        '''
        super()._control(processed_data, error)

        n = error.shape[0]
        assert(n % 2 == 0)

        # compute u as desired state time derivative for control model
        u_state = self.kp*error[:n//2] + self.kv * error[n//2:]
        u_state = np.clip(u_state, -self.u_max, self.u_max)

        if self.model.has_heading:
            u_state[-1] = np.clip(u_state[-1], -self.u_max/100.0, self.u_max/100.0)

        # invert control model to get actual action
        u = self.model.inverse(processed_data, u_state)

        return u
