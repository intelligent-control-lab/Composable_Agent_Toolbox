import numpy as np
import world
import matplotlib.pyplot as plt
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

    def reset(self):
        env_info, sensor_data = self.world.reset()
        return self.dt, env_info, sensor_data
    def step(self, controls):
        env_info, sensor_data = self.world.simulate(controls, self.dt)
        self.render(env_info)
        return self.dt, env_info, sensor_data

    def render(self, env_info):
        plt.cla()
        plt.axis([0, 100, 0, 100])
        x = []
        y = []
        for i in range(len(env_info)):
            x.append(env_info[i][0])
            y.append(env_info[i][1])
        plt.scatter(x,y,s=100)
        plt.pause(0.1)
        plt.draw()