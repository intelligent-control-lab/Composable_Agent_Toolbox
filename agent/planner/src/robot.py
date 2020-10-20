from numpy import pi
import numpy as np
from .utils import cap

class RobotProperty():
    def __init__(self):
        self.nlink = 7
        self.DH = np.array([[0,   0.333,   0, -pi/2],
                   [0,   0,   0,   pi/2],
                   [0,   0.316,   0.088,   pi/2],
                   [0,   0,   -0.088,  -pi/2],
                   [0,   0.384,   0,   pi/2],
                   [0,   0,   0.088,   pi/2],
                   [0,   0.227,   0,   0]])

        self.vmax = np.array([1,1,1,1,1,1,1])

        self.lb = np.array([-pi,-pi,-pi,-pi,-pi,-pi,-pi]);
        self.ub = np.array([pi,pi,pi,pi,pi,pi,pi]);
        self.margin = 0.01
        self.base = np.array([[0.], [0.], [0.]]) # base link offset
        # self.cap_end = np.array([[0], [0], [-0.163350038779012]])

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
        self.cap[0].p = np.array([[0, 0], [0, 0.333], [0, 0]])
        self.cap[0].r = 0.05
        
        self.cap[1].p = np.array([[0, 0], [0, 0], [0, 0.2521]])
        self.cap[1].r = 0.05
        
        self.cap[2].p = np.array([[0, 0], [0, 0], [0, 0]])
        self.cap[2].r = 0.05
        
        self.cap[3].p = np.array([[0, 0], [0, 0], [0.0639, 0.3840]])
        self.cap[3].r = 0.05
        
        self.cap[4].p = np.array([[0, 0], [0, 0], [0, 0]])
        self.cap[4].r = 0.05

        self.cap[5].p = np.array([[0, 0], [0, 0], [-0.0813, 0.1070]])
        self.cap[5].r = 0.05

        self.cap[6].p = np.array([[0, 0], [0, 0], [-0.227, 0]])
        self.cap[6].r = 0.05


