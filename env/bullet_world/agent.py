
import numpy as np

import os
import pybullet as p
import pybullet_data

import math
import random
import importlib


class FrankaPanda(object):

    def __init__(self, name, spec, collision=True):
        self.name = name
        
        self.collision = collision
        self.broadcast = {}

        self.init_joints = spec["init_joints"]
        self.base_position = spec["base_position"]

        self.requires_control = True
        
        self.vel = np.vstack(np.zeros(3))

        self.reset()

    @property
    def state(self):
        return self.pos
        joint_angles = list(map(lambda x:x[0], p.getJointStates(self.model_uid, range(7))))
        return np.vstack(np.array(joint_angles))
    
    @property
    def pos(self):
        return np.vstack(np.array(p.getLinkState(self.model_uid, 11)[0]))

    def forward(self, action, dt):
        control = action["control"]
        
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        
        control = np.maximum(control, -1)
        control = np.minimum(control,  1)

        self.vel = np.vstack(control)

        dx = control[0] * dt
        dy = control[1] * dt
        dz = control[2] * dt
        fingers = 0

        currentPose = p.getLinkState(self.model_uid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.model_uid,11,newPosition, orientation)

        p.setJointMotorControlArray(self.model_uid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses[:7])+2*[fingers])

        self.joint_state  = p.getLinkState(self.model_uid, 11)[0]
        self.finger_state = (p.getJointState(self.model_uid,9)[0], p.getJointState(self.model_uid, 10)[0])

        self.broadcast = action["broadcast"] if "broadcast" in action.keys() else {}
        
    def reset(self):
        urdfRootPath=pybullet_data.getDataPath()
        
        self.model_uid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, basePosition=self.base_position)
        
        for i in range(7):
            p.resetJointState(self.model_uid,i, self.init_joints[i])

        state_robot = p.getLinkState(self.model_uid, 11)[0]
        state_fingers = (p.getJointState(self.model_uid,9)[0], p.getJointState(self.model_uid, 10)[0])
        observation = state_robot + state_fingers

        return observation


class Ball(object):
    def __init__(self, name, spec, collision=True):
        self.name = name
        self.collision = collision
        self.broadcast = {}
        self.init_position = spec["init_position"]
        self.requires_control = True
        self.vel = np.vstack(np.zeros(3))
        self.reset()

    @property
    def state(self):
        return np.vstack([self.pos, self.vel])
    
    @property
    def pos(self):
        pos, ori = p.getBasePositionAndOrientation(self.model_uid)
        return np.vstack(pos)

    def forward(self, action, dt):
        control = action["control"]

        control = np.maximum(control, -1)
        control = np.minimum(control,  1)

        self.vel = np.vstack(control)
        
        currentPosition = self.pos
        newPosition = [currentPosition[0] + self.vel[0] * dt,
                       currentPosition[1] + self.vel[1] * dt,
                       currentPosition[2] + self.vel[2] * dt]

        ori = [0,0,0,1]
        p.resetBasePositionAndOrientation(self.model_uid, newPosition, ori)
        self.broadcast = action["broadcast"] if "broadcast" in action.keys() else {}
        
    def reset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_uid = p.loadURDF(os.path.join(dir_path, "urdfs/human.urdf"), basePosition=self.init_position)
        return self.pos

class BallGoal(object):
    def __init__(self, name, hunter, goal_list, reaching_eps, collision=False):
        self.name = name
        self.goal_list = goal_list
        self.goal_idx = 0
        self.hunter = hunter
        self.reaching_eps = reaching_eps
        self.collision = collision
        self.requires_control = False
        self.broadcast = {}
        self.reset()

    @property
    def state(self):
        return np.vstack([self.pos, self.vel])
    
    @property
    def pos(self):
        pos, ori = p.getBasePositionAndOrientation(self.model_uid)
        return np.vstack(pos)

    @property
    def vel(self):
        return np.vstack(np.zeros(3))
    
    def _set_pos(self):
        pos = self.goal_list[self.goal_idx]
        ori = [0,0,0,1]
        p.resetBasePositionAndOrientation(self.model_uid, pos, ori)

    def forward(self):
        if np.max(abs(self.pos - self.hunter.pos)) < self.reaching_eps:
            self.goal_idx = min(len(self.goal_list)-1, self.goal_idx+1)
            self._set_pos()
        
    def reset(self):
        self.goal_idx = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if "human" in self.name:
            self.model_uid = p.loadURDF(os.path.join(dir_path, "urdfs/human_goal.urdf"), basePosition=[0,0,0])    
        else:
            self.model_uid = p.loadURDF(os.path.join(dir_path, "urdfs/robot_goal.urdf"), basePosition=[0,0,0])
        self._set_pos()
        
        return self.pos

    