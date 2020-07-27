"""This is the mujoco physics engine module.

The MujocoWorld class is a wrapper between the environment and the actual physics engine. It will add agents and sensors
define by users to the physics engine, and return the simulation results. 

The World class is called by the Envrionment class.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data

import math
import numpy as np
import random
import importlib

from env.base_world.world import World
import env.flat_world.agent

class BulletWorld(World):
    metadata = {'render.modes': ['human']}

    def __init__(self, spec):
        self.spec = spec
        self.agents = {}
        self.sensor_groups = {}

    def _collect_agent_info(self):
        """Collect agent position.
        """
        agents_pos = {}
        for agent in self.agents.values():
            agents_pos[agent.name] = agent.pos
        return agents_pos
        
    def simulate(self, actions, dt):
        """One step simulation in the physics engine

        Args:
            actions: a dictionary contains the action of all agents. 
                Each action is also a dictionary, contains the control input and other information.
            dt: time separation between two steps.
        
        Returns:
            environment infomation: contains information needed by rendering and evaluator.
            measurement_groups: data of all sensors grouped by agent names.
        """
        for name, agent in self.agents.items():
            agent.forward(actions[name], dt)
        
        p.setTimeStep(dt)
        p.stepSimulation()

        env_info = self._collect_agent_info()
        measurement_groups = self._collect_sensor_data()
        
        return env_info, measurement_groups
        
    def reset(self):
        p.resetSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)
        urdfRootPath=pybullet_data.getDataPath()
        
        self.planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        self.trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        
        env_info = self._collect_agent_info()
        agent_sensor_data = self._collect_sensor_data()
        return env_info, agent_sensor_data
     