"""This is the physics engine module. Will be replaced by other physics engine in the future.

The World class is a simple physics engine. It will instantiate agents and sensors
define by users, and simulate the interactions. 

The World class is called by the Envrionment class.
"""

import numpy as np
import world.sensor
import world.agent
import importlib
class World(object):
    def __init__(self, spec):
        self.spec = spec
        self.agents = []
        self.sensor_groups = {}
        # sensors[i] : [sensor1_data, sensor2_data, ...] is the i-th agent's all sensor data.
        
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
        AgentClass = getattr(importlib.import_module("world.agent"), agent_env_spec["type"])
        instance = AgentClass(comp_agent.name, agent_env_spec['init_x'])

        self.agents.append(instance)
        agent_sensors = []
        for j in range(len(comp_agent.sensors)):
            agent_sensors.append(self._add_sensor(instance, comp_agent.sensors[j].spec))
        self.sensor_groups[comp_agent.name] = agent_sensors
        
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
        return SensorClass(agent, self.agents, spec)

    def reset(self):
        """Reset the physics engine.

        Nothing is wiped. Only data is reset.
        """
        self.cache = None
        env_info = self._collect_agent_pos()
        agent_sensor_data = self._collect_sensor_data()
        return env_info, agent_sensor_data
    
    def _collect_sensor_data(self):
        """Collect data of all sensors.
        
        Call measure() of every sensor and return measurements by groups.

        Return:
            A dict mapping agent_name to a measurement group.
            Each group is also a dict that mapping sensor_name to measurements, e.g.
            {
                'robot':{'state_sensor': {'pos': array([[0], [0]]), 'vel': array([[0], [1]])}, 
                         'obstacle_sensor': {'human': array([[0], [10]])}},
                'human':{'state_sensor': (array([[0],[10]]), array([[0],[0]]))}
            }
        """
        measurement_groups = {}
        for name, group in self.sensor_groups.items():
            measurements = {}
            for sensor in group:
                measurements[sensor.alias] = sensor.measure()
            measurement_groups[name] = measurements
        return measurement_groups

    def _collect_agent_pos(self):
        """Collect agent position.
        """
        agents_pos = []
        for i, agent in enumerate(self.agents):
            agents_pos.append(agent.pos)
        return agents_pos

    def simulate(self, controls, dt):
        """One step simulation in the physics engine

        Args:
            controls: the action taken by the computational agents.
            dt: time separation between two steps.
        
        Returns:
            environment infomation: contains information needed by rendering and evaluator.
            measurement_groups: data of all sensors grouped by agent names.
        """
        for i,agent in enumerate(self.agents):
            agent.forward(controls[i], dt)

        env_info = self._collect_agent_pos()
        measurement_groups = self._collect_sensor_data()
        
        return env_info, measurement_groups
