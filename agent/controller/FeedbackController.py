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
    def compute_error(self, processed_data: dict, goal: np.ndarray) -> np.ndarray:
        '''
            Given estimated data and goal, get error in goal space
        '''

        # column vector
        assert(len(goal.shape) == 2 and goal.shape[1] == 1)

        pass

    @abstractmethod
    def inverse_kinematics(self, cart: np.ndarray, param: dict) -> np.ndarray:
        '''
            Convert to state space
        '''
        pass

    @abstractmethod
    def control(self, error: np.ndarray) -> np.ndarray:
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

        e = self.compute_error(processed_data=processed_data, goal=goal)

        if goal_type == GoalType.CARTESIAN:
            e = self.inverse_kinematics(cart=e, param=None)
        
        u = self.control(error=e)

        return u


class NaiveFeedbackController(FeedbackController):

    def __init__(self, spec, model):
        super().__init__(model)

        # weights
        self.kp = spec["kp"]
        self.kv = spec["kv"]
    
    def compute_error(self, processed_data, goal):
        '''
            Cartesian error
        '''
        super().compute_error(processed_data, goal)
                
        e = goal - np.vstack([
                processed_data["cartesian_sensor_est"]["pos"],
                processed_data["cartesian_sensor_est"]["vel"]])
        
        return e

    def inverse_kinematics(self, cart, param):
        super().inverse_kinematics(cart, param)

        return cart

    def control(self, error):
        '''
            P control on both pos and vel
        '''
        super().control(error)

        n = error.shape[0]
        assert(n % 2 == 0)
        u = self.kp*error[:n//2] + self.kv * error[n//2:]

        return u
