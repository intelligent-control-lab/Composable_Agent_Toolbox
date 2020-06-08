import numpy as np
import world
class Environment(object):
    def __init__(self, env_spec, agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        self.dt = env_spec['dt']
        self.world = world.World(env_spec)
        for i in range(len(agents)):
            self.world.add_agent(agents[i], env_spec['agent_env_spec'][i])
            for j in range(len(agents[i].sensors)):
                self.world.add_sensor(agents[i].sensors[j].spec)

    def reset(self):
        env_info, sensor_data = self.world.reset()
        return env_info, sensor_data
    def step(self, controls):
        env_info, sensor_data = self.world.simulate(controls, self.dt)
        return env_info, sensor_data
    def render(self):
        pass