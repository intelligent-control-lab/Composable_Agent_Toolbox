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
    def ComputeError(self, processed_data: dict, goal: np.ndarray) -> np.ndarray:
        '''
            Given estimated data and goal, get error in goal space
        '''

        # column vector
        assert(len(goal.shape) == 2 and goal.shape[1] == 1)

        pass

    @abstractmethod
    def InverseKinematics(self, cart: np.ndarray, param: dict) -> np.ndarray:
        '''
            Convert to state space
        '''
        pass

    @abstractmethod
    def Control(self, error: np.ndarray) -> np.ndarray:
        '''
            Can be model inverse or other control algo
        '''
        pass
    
    def __call__(self,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''

        e = self.ComputeError(processed_data=processed_data, goal=goal)

        if goal_type == GoalType.CARTESIAN:
            e = self.InverseKinematics(cart=e, param=None)
        
        u = self.Control(error=e)

        return u


class NaiveFeedbackController(FeedbackController):

    def __init__(self, model, kp, kv):
        super().__init__(model)

        # weights
        self.kp = kp
        self.kv = kv
    
    def ComputeError(self, processed_data, goal):
        '''
            Cartesian error
        '''
        super().ComputeError(processed_data, goal)
                
        e = goal - np.vstack([
                processed_data["cartesian_sensor_est"]["pos"],
                processed_data["cartesian_sensor_est"]["vel"]])
        
        return e

    def InverseKinematics(self, cart, param):
        super().InverseKinematics(cart, param)

        return cart

    def Control(self, error):
        '''
            P control on both pos and vel
        '''
        super().Control(error)

        n = error.shape[0]
        assert(n % 2 == 0)
        u = self.kp*error[:n//2] + self.kv * error[n//2:]

        return u
