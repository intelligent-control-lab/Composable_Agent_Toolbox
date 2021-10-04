import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../../'))

import numpy as np
from abc import ABC, abstractmethod
from utils import GoalType

from numpy.matlib import repmat
from numpy import zeros, eye, matrix

class KinematicDynamicModel():
    """This is the base class for all robot dynamic models. 
    We assume the models are all in the form:

    :math:`X' = A * X +  B * u`

    :math:`\dot X  =  fx + fu * u`
    
    Because

    :math:`X' = X + \dot X * dT`

    Then

    :math:`fx = (A - I) / dT`

    :math:`fu = B / dT`

    We just need to specify A and B to define different dynamic models.

    There are two major phases in the control circle, update and move. In the update phase, the robot will update its information based on the environment. And in the move phase, the robot will execute control input.
    """

    def __init__(self, spec):
        """This function initilize the robot.
        
        Args:
            init_state (list): the init state of the robot, for example [x, y, vx, vy]
            agent (MobileAgent()): the algorithm that controls this robot.
            dT (float): the seperation between two control output
            auto (bool): whether this robot is autonomous, if not, it is control by user input like mouse.
            is_2D (bool): whether this model is a 2D model, which means it can only move on the groud plane.

        """

        self.dT = spec["dT"]
        self.control_input_dim = spec["control_input_dim"]

    def fx(self, x):
        """
        This function calculate fx from A,
        Because
        X' = X + dot_X * dT
        Then
        fx = (A - I) / dT
        """
        return (self.A() - np.eye(np.shape(x)[0])) / self.dT * x
    def fu(self):
        """
        This function calculate fu from B,
        Because
        X' = X + dot_X * dT
        Then
        fu = B / dT
        """
        return self.B() / self.dT

    def update_m(self, Mh):
        """Update the nearest cartesian point on self to obstacle. 
        Args:
            Mh (ndarray): 6*1 array, cartesian postion and velocity of the obstacle.
        """
        self.m = self.get_closest_X(Mh)

# The following functions are required to fill up for new models.

    def get_P(self):
        """
        Return position in the Cartisian space.
        """
        pass
    def get_V(self):
        """
        Return velocity in the Cartisian space.
        """
        pass
    
    def A(self):
        """
        Transition matrix A as explained in the class definition.
        """
        pass
    def B(self):
        """
        Transition matrix B as explained in the class definition.
        """
        pass
    def get_closest_X(self, Mh):
        """
        Update the corresponding state of the nearest cartesian point on self to obstacle. 
        
        Args:
            Mh (ndarray): 6*1 array, cartesian postion and velocity of the obstacle.
        """
        pass
    def jacobian(self): # p closest point p X
        """ dM / dX, the derivative of the nearest cartesian point to robot state.
        """
        pass
    def estimate_state(self):
        """
        State estimater caller.
        """
        pass
    def u_ref(self):
        """
        Reference control input.
        """
        pass


class BallModel(KinematicDynamicModel):

    """
    This the 2D ball model. The robot can move to any direction on the plane.
    """
    
    def __init__(self, spec):
        super().__init__(spec)
 
    def get_P(self):
        return np.vstack(self.x[[0,1]])
    
    def get_V(self):
        return np.vstack(self.x[[2,3]])

    def get_PV(self):
        return np.vstack([self.get_P(), self.get_V()])

    def get_closest_X(self, Xh):
        self.m = self.get_PV()
        return self.m

    def A(self):
        A = matrix(eye(4,4))
        A[0,2] = self.dT
        A[1,3] = self.dT
        return A

    def B(self):
        B = matrix(zeros((4,2)))
        B[0,0] = self.dT**2/2
        B[1,1] = self.dT**2/2
        B[2,0] = self.dT
        B[3,1] = self.dT
        return B

    def jacobian(self): # p closest point p X
        ret = np.eye(4)
        return ret