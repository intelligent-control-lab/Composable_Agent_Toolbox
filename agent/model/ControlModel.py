from abc import abstractmethod
import numpy as np
import math
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
        self.has_heading = False

    @property
    def state_component(self):
        return self._state_component

    @abstractmethod
    def _compute_error(self, processed_data: dict, goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:

        pass

    def compute_error(self, processed_data: dict, goal: np.ndarray,
        goal_type: GoalType, state_dimension: int) -> np.ndarray:
        '''
            goal: from planner traj
            state_dimension: defined by planner

            Assume that controller always uses full dimension of planned trajectory
            But can ignore derivatives, e.g., goal = [pos; vel], and control tracks [pos]
            pos/vel/etc. contains all state dimensions considered in planning model

            Derived model will determine what info to use from processed data.
        '''

        # useful part of goal
        g = goal[:len(self._state_component)*state_dimension]

        return self._compute_error(
            processed_data=processed_data,
            goal=g,
            goal_type=goal_type)
    
    @abstractmethod
    def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:
        '''
            compute action u from xdot
        '''
        pass

class BallModel(ControlModel):
    
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
    
    def _compute_error(self, processed_data: dict,
        goal: np.ndarray, goal_type: GoalType) -> np.ndarray:
        '''
            Cartesian error.
        '''

        e = goal - np.vstack([
                processed_data["cartesian_sensor_est"]["pos"],
                processed_data["cartesian_sensor_est"]["vel"]])
        
        return e
    
    def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:

        super().inverse(processed_data, xdot_desired)

        return xdot_desired

class UnicycleModel(ControlModel):
    '''
        state _x is [x, y, t]
        action is [v, w], v -> velocity, w -> heading velocity
        [xdot, ydot, tdot] = [vcost, vsint, w]
    '''
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)

        self.has_heading = True

    def _compute_error(self, processed_data: dict, goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        
        xref, yref = goal.reshape(-1)
        x, y, t = processed_data["cartesian_sensor_est"][self.state_component[0]].reshape(-1)

        xerr = xref - x
        yerr = yref - y
        terr = math.atan2( yerr, xerr ) - t
        terr = math.atan2(math.sin(terr), math.cos(terr)) # map to +/- pi

        return np.array([xerr, yerr, terr, 0.0, 0.0, 0.0]).reshape(-1, 1)

    def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:
        
        _, _, t = processed_data["cartesian_sensor_est"][self.state_component[0]].reshape(-1)
        
        A = np.array(
            [
                [math.cos(t), 0.0],
                [math.sin(t), 0.0],
                [0.0        , 1.0]
            ]
        )
        
        b = xdot_desired

        u = np.linalg.lstsq(a=A, b=b)[0] # [v, w]

        return u

