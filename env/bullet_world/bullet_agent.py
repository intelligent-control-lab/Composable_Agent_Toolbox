
import numpy as np

import os
import pybullet as p
import pybullet_data

import math
import random
import importlib

    
class FrankaPanda():

    def __init__(self, name, spec, collision=True):
        self.name = name
        
        self.collision = collision
        self.broadcast = {}

        self.init_joints = spec["init_joints"]
        self.base_position = spec["base_position"]

        self.reset()

    @property
    def state(self):
        return p.getLinkState(self.pandaUid, 11)
    
    def pos(self):
        return p.getLinkState(self.pandaUid, 11)[0]

    def vel(self):
        return p.getLinkState(self.pandaUid, 11)[0]
    
    def forward(self, action, dt):
        control = action["control"]
        
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])

        dx = control[0] * dt
        dy = control[1] * dt
        dz = control[2] * dt
        fingers = control[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses[:7])+2*[fingers])

        self.joint_state  = p.getLinkState(self.pandaUid, 11)[0]
        self.finger_state = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        
    def reset(self):
        urdfRootPath=pybullet_data.getDataPath()
        
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, basePosition=self.base_position)
        
        for i in range(7):
            p.resetJointState(self.pandaUid,i, self.init_joints[i])

        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # rendering's back on again
        return observation