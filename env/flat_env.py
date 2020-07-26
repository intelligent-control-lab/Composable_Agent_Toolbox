import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib

class FlatEnv(object):
    def __init__(self, env_spec, agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        self.dt = env_spec['dt']
        WorldClass = getattr(importlib.import_module("env.flat_world"), env_spec["world"]["type"])
        self.world = WorldClass(env_spec["world"]["spec"])
        for i in range(len(agents)):
            self.world.add_agent(agents[i], env_spec['agent_env_spec'][agents[i].name])

    def reset(self):
        env_info, sensor_data = self.world.reset()
        return self.dt, env_info, sensor_data

    def step(self, actions):
        env_info, sensor_data = self.world.simulate(actions, self.dt)
        self.render(env_info)
        return self.dt, env_info, sensor_data

    def render(self, env_info):
        plt.cla()
        plt.axis([0, 100, 0, 100])
        x = []
        y = []
        for name, pos in env_info.items():
            if 'goal' not in name:
                x.append(pos[0])
                y.append(pos[1])
        for name, pos in env_info.items():
            if 'goal' in name:
                x.append(pos[0])
                y.append(pos[1])
        
        cs = ['#ff0000', '#0000ff', '#ff5500', '#3399ff']
        plt.scatter(x,y,s=100, color=cs[:len(x)])
        plt.pause(0.1)
        plt.draw()
