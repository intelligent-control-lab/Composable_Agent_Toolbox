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
        # self._state_component = spec['state_component']
        self.has_heading = False

        self.feat = self._feat
        self.w = None

    @abstractmethod
    def _feat(self, processed_data: dict, goal: np.ndarray) -> np.ndarray:
        '''
            control error = w @ _feat
        '''

        pass

    def get_goal(self, goal: np.ndarray) -> np.ndarray:
        raise NotImplemented
    
        # # useful part of goal
        # return goal[:4]

    def compute_error(self, processed_data: dict, goal: np.ndarray,
        goal_type: GoalType, state_dimension: int) -> np.ndarray:
        '''
            goal: from planner traj
            state_dimension: defined by planner

            Assume that controller always uses full dimension of planned trajectory
            But can ignore derivatives, e.g., goal = [pos; vel], and control tracks [pos]
            pos/vel/etc. contains all state dimensions considered in planning model

            Derived model will determine what info to use from processed data.

            Output is error in control space
        '''
        
        g = self.get_goal(goal)
        return self.w @ self.feat(processed_data, g)
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented
        
    # @abstractmethod
    # def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:
    #     '''
    #         compute action u from xdot
    #     '''
    #     pass

class BallModel(ControlModel):
    
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.w = np.eye(4)

    def get_goal(self, goal: np.ndarray) -> np.ndarray:
        return goal[:4]

    def _feat(self, processed_data: dict, goal: np.ndarray) -> np.ndarray:
        return goal - np.vstack([
                processed_data["cartesian_sensor_est"]["pos"],
                processed_data["cartesian_sensor_est"]["vel"]])
    
    # dx = [vx, vy, ax, ay]
    # fx = [vx, vy, 0, 0]
    # fu = [[1,0],[0,1]] @ [ax, ay]
    def fx(self, x: np.ndarray) -> np.ndarray:
        fx = np.vstack([x[2], x[3], 0, 0])
        return  fx
    
    def fu(self, x:np.ndarray) -> np.ndarray:
        fu = np.zeros((4,2))
        fu[2,0] = 1
        fu[3,1] = 1
        return fu
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        # p_cartesian_p_state
        return np.eye(4)
    
    # def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:

    #     super().inverse(processed_data, xdot_desired)

    #     return xdot_desired

class UnicycleModel(ControlModel):
    '''
        state _x is [x, y, t]
        action is [v, w], v -> velocity, w -> heading velocity
        [xdot, ydot, tdot] = [vcost, vsint, w]
    '''
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)

        self.has_heading = True
        self.w = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1.5]
        ])

    def get_goal(self, goal: np.ndarray) -> np.ndarray:
        return goal[:2]
    
    def _feat(self, processed_data: dict, goal: np.ndarray) -> np.ndarray:
        
        xref, yref = goal.reshape(-1)
        # x, y, t = processed_data["cartesian_sensor_est"][self.state_component[0]].reshape(-1)
        # print(processed_data["state_sensor_est"])
        # print(processed_data["state_sensor_est"]['state'][:3].reshape(-1))
        x, y, t = processed_data["state_sensor_est"]['state'][:3].reshape(-1)

        xerr = xref - x
        yerr = yref - y
        tg = math.atan2( yerr, xerr )

        # terr = math.atan2(math.sin(terr), math.cos(terr)) # map to +/- pi

        feat = np.zeros((3, 1))
        feat[0, 0] = math.sqrt(xerr**2 + yerr**2)
        feat[1, 0] = math.cos(tg - t) * math.sqrt(xerr**2 + yerr**2)
        feat[2, 0] = math.atan2(math.sin(tg - t), math.cos(tg - t))

        return feat

    # dx = fx + fu * u
    # vx, vy, vt, v cos(t), v sin(t), w
    # fx = [vx, vy, vt, 0, 0, 0].T
    # fu * u = [[0,0], [0,0], [0,0], [cos(t),0], [sin(t),0], [0 ,1]] * [v; w]
    def fx(self, x: np.ndarray) -> np.ndarray:
        # [x,y,t, vx, vy, vt]
        # 
        fx = np.vstack([x[3], x[4], x[5], 0, 0, 0])
        return  fx
    def fu(self, x:np.ndarray) -> np.ndarray:
        fu = np.zeros((6, 2))
        fu[3, 1] = np.cos(x[2])
        fu[4, 1] = np.sin(x[2])
        fu[5, 1] = 1
        return fu
        
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        # p_cartesian_p_state
        p_ce_p_xe = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0]])
        
        return p_ce_p_xe
        
        # cartesian = [x, y, vx, vy]
        # state = [x, y, t, vx, vy, vt]
        # dot_c = p_c_p_x * dot_x
        
        
        
    # def inverse(self, processed_data: dict, xdot_desired: np.ndarray) -> np.ndarray:
        
    #     _, _, t = processed_data["cartesian_sensor_est"][self.state_component[0]].reshape(-1)
        
    #     A = np.array(
    #         [
    #             [math.cos(t), 0.0],
    #             [math.sin(t), 0.0],
    #             [0.0        , 1.0]
    #         ]
    #     )
        
    #     b = xdot_desired

    #     u = np.linalg.lstsq(a=A, b=b)[0] # [v, w]

    #     return u

