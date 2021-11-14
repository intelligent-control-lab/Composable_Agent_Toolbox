from abc import abstractmethod
import numpy as np
from abc import ABC, abstractmethod

import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))
from utils import GoalType

class ControlModel(ABC):
    '''
        xdot = f(x) + g(x)*u
    '''
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self._state_component = spec['state_component']

    @property
    def state_component(self):
        return self._state_component

    @abstractmethod
    def _compute_error(self,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType
    ) -> np.ndarray:

        pass

    def compute_error(self,
        processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType,
        state_dimension: int
    ):
        '''
            goal: from planner traj
            state_dimension: defined by task, also consistent with planner
        '''

        # useful part of goal
        g = goal[:len(self._state_component)*state_dimension]

        return self._compute_error(
            processed_data=processed_data,
            goal=g,
            goal_type=goal_type)
    
    @abstractmethod
    def inverse(self,
        processed_data: dict,
        xdot_desired: np.ndarray
    ) -> np.ndarray:
        '''
            compute action u from xdot
        '''
        pass

class BallModel(ControlModel):
    
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
    
    # todo for unicycle, convert xy error to xytheta error
    def _compute_error(
        self, processed_data: dict,
        goal: np.ndarray,
        goal_type: GoalType
    ) -> np.ndarray:
        '''
            Cartesian error.
        '''
        super()._compute_error(processed_data, goal, goal_type)
        
        e = goal - np.vstack([
                processed_data["cartesian_sensor_est"]["pos"],
                processed_data["cartesian_sensor_est"]["vel"]])
        
        return e
    
    # todo for unicycle, get v, w by solving linear system
    def inverse(self,
        processed_data: dict,
        xdot_desired: np.ndarray
    ) -> np.ndarray:

        super().inverse(processed_data, xdot_desired)

        return xdot_desired
