import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib

class FlatEnv(object):
    def __init__(self, env_spec, comp_agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        self.dt = env_spec['dt']
        WorldClass = getattr(importlib.import_module("env.flat_world"), env_spec["world"]["type"])
        self.world = WorldClass(env_spec["world"]["spec"])
        self.env_spec = env_spec
        self.comp_agents = comp_agents
        self.reset()

    def reset(self):
        self.world.reset()
        for i in range(len(self.comp_agents)):
            self.world.add_agent(self.comp_agents[i], self.env_spec['agent_env_spec'][self.comp_agents[i].name])
        env_info, sensor_data = self.world.measure()
        return self.dt, env_info, sensor_data

    def step(self, actions):
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
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
        # plt.plot(human_traj[:,0],human_traj[:,1])
        # plt.plot(robot_traj[:,0],robot_traj[:,1])
        plt.pause(0.00001)
        plt.draw()
