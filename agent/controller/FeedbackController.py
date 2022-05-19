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

        # goal -> control space error (pos + vel)
        # e.g., for unicycle, convert x/y coord (planned goal) to error in vel/heading
        e = self.model.compute_error(
            processed_data=processed_data, goal=goal,
            goal_type=goal_type, state_dimension=state_dimension)

        # control space error -> action
        # e.g., for unicycle, compute vel/heading from vel/heading error
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

        # print(f'pos error:\n{error[:2]}')
        # print(f'vel error:\n{error[2:]}')

        # compute u as desired state time derivative for control model
        u = self.kp*error[:n//2] + self.kv * error[n//2:]
        # print(f"u:{u}")
        for i in range(u.shape[0]):
            u[i] = np.clip(u[i], -self.u_max[i], self.u_max[i])
        # print(f"control:\n{u}")
        return u
