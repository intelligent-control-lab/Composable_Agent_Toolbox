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

        raise NotImplementedError # TODO delete this line

        # TODO compute the following terms
        phi         = None
        p_phi_p_ce  = None
        p_phi_p_co  = None
        
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

        raise NotImplementedError # TODO delete this line

        # TODO compute the following terms
        phi = None
        u = None

        return phi, u

