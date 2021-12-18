import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from utils import GoalType
import cvxopt

class SafeController(ABC):
    def __init__(self, spec, model):

        self.model = model
        self.u_max = np.vstack(spec["u_max"])

    @abstractmethod
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        pass

class UnsafeController(SafeController):
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        return u_ref

class EnergyFunctionController(SafeController):
    """
    Energy function based safe controllers

    Attributes:
        _name
        _spec
        _model
    """
    def __init__(self, spec, model):
        self._spec = spec
        self._model = model
        self.d_min = spec['d_min']
        self.eta = spec['eta']
        self.k_v = spec['k_v']

    def phi_and_derivatives(self, dt, ce, co):
        """
        ce: cartesian position of ego
        co: cartesian position of an obstacle
        """
        n = np.shape(ce)[0]//2

        dp = np.vstack(ce[:n] - co[:n])
        dv = np.vstack(ce[n:] - co[n:])

        d     = max(np.linalg.norm(dp), 1e-3)
        dot_d = dp.T @ dv / d

        phi = self.d_min**2 - d**2 - self.k_v * dot_d + self.eta * dt

        p_phi_p_d = -2 * d
        p_phi_p_dot_d = - self.k_v

        p_d_p_ce = np.vstack([dp / d, np.zeros((n,1))])
        p_d_p_co = -p_d_p_ce

        p_dot_d_p_dp = dv / d - np.asscalar(dp.T @ dv) * dp / (d**3)
        p_dot_d_p_dv = dp / d

        p_dp_p_ce = np.hstack([np.eye(n), np.zeros((n,n))])
        p_dp_p_co = -p_dp_p_ce

        p_dv_p_ce = np.hstack([np.zeros((n,n)), np.eye(n)])
        p_dv_p_co = -p_dv_p_ce

        p_dot_d_p_ce = p_dp_p_ce.T @ p_dot_d_p_dp + p_dv_p_ce.T @ p_dot_d_p_dv
        p_dot_d_p_co = p_dp_p_co.T @ p_dot_d_p_dp + p_dv_p_co.T @ p_dot_d_p_dv

        p_phi_p_ce = p_phi_p_d * p_d_p_ce + p_phi_p_dot_d * p_dot_d_p_ce
        p_phi_p_co = p_phi_p_d * p_d_p_co + p_phi_p_dot_d * p_dot_d_p_co
        
        return phi, p_phi_p_ce, p_phi_p_co
    
    @abstractmethod
    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.
        """
        pass
    
    def __call__(self,
        dt: float,
        processed_data: dict,
        u_ref: np.ndarray,
        goal: np.ndarray,
        goal_type: GoalType) -> np.ndarray:
        '''
            Driver procedure. Do not change
        '''
        us = []

        for obs in processed_data["obstacle_sensor_est"]:
            phi, u = self.safe_control(u_ref, obs, dt, processed_data)
            us.append((phi, u))
        
        sorted(us, key=lambda x:x[0], reverse=True) # larger phi first
        
        return us[0][1] # adopt the control that avoids the most dangerous collision.


class SafeSetController(EnergyFunctionController):
    def __init__(self, spec, model):
        super().__init__(spec, model)
        self._name = 'safe_set'

    def safe_control(self, u_ref, obs, dt, processed_data):
        """ Compute the safe control between ego and an obstacle.

        Safe set compute u by solving the following optimization:
        min || u - u_ref ||, 
        s.t.  dot_phi < eta  or  phi > 0 (eta is the safety margin used in phi)

        => p_phi_p_xe.T * dot_xe        + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * (fx + fu * u) + p_phi_p_co.T * dot_co < eta
        => p_phi_p_xe.T * fu * u < eta - p_phi_p_xe.T * fx - p_phi_p_co.T * dot_co

        """
        ce = np.vstack([processed_data["cartesian_sensor_est"]["pos"], processed_data["cartesian_sensor_est"]["vel"]])  # ce: cartesian state of ego
        co = np.vstack([processed_data["obstacle_sensor_est"][obs]["rel_pos"], processed_data["obstacle_sensor_est"][obs]["rel_vel"]]) + ce  # co: cartesian state of the obstacle
        
        x =  np.vstack(processed_data["state_sensor_est"]["state"])

        n = np.shape(ce)[0]//2

        # It will be better if we have an estimation of the acceleration of the obstacle
        dot_co = np.vstack([co[n:], np.zeros((n,1))])

        phi, p_phi_p_ce, p_phi_p_co = self.phi_and_derivatives(dt, ce, co)

        p_ce_p_xe = self._model.jacobian(x)
        # dot_x = fx + fu * u
        fx = self._model.fx(x)
        fu = self._model.fu(x)

        p_phi_p_xe = p_ce_p_xe.T @ p_phi_p_ce

        L = p_phi_p_xe.T @ fu
        S = -self.eta - p_phi_p_xe.T @ fx - p_phi_p_co.T @ dot_co
        
        u = u_ref

        if phi <= 0 or np.asscalar(L @ u_ref) < np.asscalar(S):
            u = u_ref
        else:
            u = u_ref - (np.asscalar(L @ u_ref - S) * L.T / np.asscalar(L @ L.T))

        return phi, u

