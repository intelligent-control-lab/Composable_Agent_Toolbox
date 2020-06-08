import numpy as np
#Suqin
class Controller(object):
    def __init__(self, spec, model):
        self.spec = spec
        self.model = model

    def control(self, dt, x, goal_x):
        traj_c = self.traj_control(dt, x, goal_x)
        safe_c = self.safe_control(dt, x, goal_x)
        return self.merge(traj_c, safe_c)
    
    def traj_control(self, dt, x, goal_x):
        u = np.ones(self.model.u_shape)
        return u
    
    def safe_control(self, dt, x, goal_x):
        u = np.ones(self.model.u_shape)
        return u

    def merge(self, traj_c, safe_c):
        return traj_c + safe_c