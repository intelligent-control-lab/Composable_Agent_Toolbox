import numpy as np
import world.sensor
import world.agent

class World(object):
    def __init__(self, spec):
        self.spec = spec
        self.agents = []
        self.sensors = []
        # sensors[i] : [sensor1_data, sensor2_data, ...] is the i-th agent's all sensor data.
        
    def add_agent(self, agent, agent_env_spec):

        instance = world.agent.BB8(agent_env_spec['init_x'])
        self.agents.append(instance)
        agent_sensors = []
        for j in range(len(agent.sensors)):
            agent_sensors.append(self._add_sensor(instance, agent.sensors[j].spec))
        
        self.sensors.append(agent_sensors)
        
    def _add_sensor(self, agent, spec):
        return world.sensor.PVSensor(agent, spec)

    def reset(self):
        self.cache = None
        env_info = self._collect_agent_pos()
        agent_sensor_data = self._collect_sensor_data()
        return env_info, agent_sensor_data
    
    def _collect_sensor_data(self):
        agent_sensor_data = []
        for sensors in self.sensors:
            sensors_data = []
            for sensor in sensors:
                sensors_data.append(sensor.measure())
            agent_sensor_data.append(sensors_data)
        return agent_sensor_data

    def _collect_agent_pos(self):
        agents_pos = []
        for i, agent in enumerate(self.agents):
            agents_pos.append(agent.pos)
        return agents_pos

    def simulate(self, controls, dt):
        for i,agent in enumerate(self.agents):
            agent.forward(controls[i], dt)

        env_info = self._collect_agent_pos()
        agent_sensor_data = self._collect_sensor_data()
        
        return env_info, agent_sensor_data
