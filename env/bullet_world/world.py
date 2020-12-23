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
from env.flat_world.world import FlatReachingWorld
import env.bullet_world.agent

class BulletWorld(FlatReachingWorld):
    metadata = {'render.modes': ['human']}

    def __init__(self, spec):
        super().__init__(spec)

    def add_agent(self, comp_agent, agent_env_spec):
        """Instantiate an agent in the environment based on the computational agent and specs.

        This function instantiate a user defined agents in the physical world. 
        And sensors attached to this agent will also be instantiated by calling  _add_sensor.

        Args:
            comp_agent: the computational agent object which decides the behavior of the agent.
                This object also constains the specs of the agent, such as name, sensor types, etc.
            agent_env_spec: initial position of the physical agent in the simulation environment, etc.
        """
        
        # Instantiate an agent by the name(string, user given) of the agent class (e.g. "Unicycle").
        # "getattr" is not a good practice, but I didn't find a better way to do it.
        AgentClass = getattr(importlib.import_module("..agent", __name__), agent_env_spec["type"])
        agent = AgentClass(comp_agent.name, agent_env_spec["spec"])

        self.agents[agent.name] = agent
        agent_sensors = []
        for alias in comp_agent.sensors.keys():
            agent_sensors.append(self._add_sensor(agent, comp_agent.sensors[alias].spec))
        self.sensor_groups[comp_agent.name] = agent_sensors

        # print("self.sensor_groups[comp_agent.name]")
        # print("comp_agent.name")
        # print(self.sensor_groups[comp_agent.name])

        goal_agent = env.bullet_world.agent.BallGoal(agent.name+"_goal", agent, self.agent_goal_lists[agent.name], self.reaching_eps)
        agent.goal = goal_agent
        self.agents[goal_agent.name] = goal_agent

        return agent  # just in case subclasses call this function and need the agent handle.
        
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
            if agent.requires_control:
                agent.forward(actions[name], dt)
            else:
                agent.forward()
        
        p.setTimeStep(dt)
        p.stepSimulation()

    def reset(self):

        self.agents = {}
        self.sensor_groups = {}
        
        p.resetSimulation()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)
        urdfRootPath=pybullet_data.getDataPath()
        
        self.planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        # self.trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        # state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        # self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        