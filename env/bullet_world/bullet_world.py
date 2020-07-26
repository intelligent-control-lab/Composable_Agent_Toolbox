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

class BulletWorld():
    metadata = {'render.modes': ['human']}

    def __init__(self, spec):
        self.spec = spec
        self.agents = {}
        self.sensor_groups = {}
        
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
        AgentClass = getattr(importlib.import_module("env.bullet_world.bullet_agent"), agent_env_spec["type"])
        agent = AgentClass(comp_agent.name, agent_env_spec['spec'])

        self.agents[agent.name] = agent
        agent_sensors = []
        for alias in comp_agent.sensors.keys():
            agent_sensors.append(self._add_sensor(agent, comp_agent.sensors[alias].spec))
        self.sensor_groups[comp_agent.name] = agent_sensors

        return agent  # just in case subclasses call this function and need the agent handle.
        
    def _add_sensor(self, agent, spec):
        """Instantiate a sensor in the physics simulation and return it.
        
        Args:
            agent: the physical agent that the sensor attched on. 
                The agent can be NULL in case the sensor is a global observer and is not attached to any agent.
            spec: sensor specs given by the computational agent object.
        
        Return:
            The instantiated sensor.
        """
        SensorClass = getattr(importlib.import_module("world.sensor"), spec["type"])
        return SensorClass(agent, self.agents, spec["spec"])

    def _collect_sensor_data(self):
        """Collect data of all sensors.
        
        Call measure() of every sensor and return measurements by groups.

        Return:
            A dict mapping agent_name to a measurement group.
            Each group is also a dict that mapping sensor_name to measurements, e.g.
            {
                'robot':{'cartesian_sensor': {'pos': array([[0], [0]]), 'vel': array([[0], [1]])}, 
                         'obstacle_sensor': {'human': array([[0], [10]])}},
                'human':{'cartesian_sensor': (array([[0],[10]]), array([[0],[0]]))}
            }
        """
        measurement_groups = {}
        for name, group in self.sensor_groups.items():
            measurements = {}
            for sensor in group:
                measurements[sensor.alias] = sensor.measure()
            measurement_groups[name] = measurements
        return measurement_groups

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
        
    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()
