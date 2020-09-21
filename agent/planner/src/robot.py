from numpy import pi
import numpy as np
from .utils import cap

class RobotProperty():
    def __init__(self):
        self.nlink = 6
        self.DH = np.array([[0, 0.281, 0.145, -pi/2],
                   [-pi/2, 0, 0.87, 0],
                   [0, 0, 0.21, -pi/2],
                   [0, 1.025, 0, pi/2],
                   [ 0, 0, 0, -pi/2],
                   [0, 0.175, 0, 0]])

        self.lb = np.array([-pi,-pi,-pi,-pi,-pi,-pi]);
        self.ub = np.array([pi,pi,pi,pi,pi,pi]);

        self.base = np.array([[0.], [0.], [0.259]]) # base link offset
        self.cap_end = np.array([[0], [0], [-0.163350038779012]])

        # robot capsule for each link. 
        # Current capsule is not specified. 
        # End-effector measured offset, with respect to last coorfinate
        self.cap = []
        self.define_capsule()


    def define_capsule(self):
        for i in range(self.nlink):
            tmp = cap()
            self.cap.append(tmp)
        # set the capsules
        self.cap[0].p = np.array([[-0.145, -0.145], [0.105, 0.105], [0, 0]])
        self.cap[0].r = 0.385
        
        self.cap[1].p = np.array([[-0.87, 0], [0, 0], [-0.1945, -0.1945]])
        self.cap[1].r = 0.195
        
        self.cap[2].p = np.array([[-0.02, -0.09], [0.073, 0.073], [0.115, 0.115]])
        self.cap[2].r = 0.33
        
        self.cap[3].p = np.array([[0, 0], [-0.65, 0], [-0.0235, -0.0235]])
        self.cap[3].r = 0.115
        
        self.cap[4].p = np.array([[0, 0], [0.0145, 0.0145], [0.025, 0.025]])
        self.cap[4].r = 0.15

        self.cap[5].p = np.array([[0.00252978728478826, 0.000390496336481267], [6.28378116607958e-10,1.00300828261106e-10], [-0.170767309373314,-0.0344384157898974]])
        self.cap[5].r = 0.03

