import multiprocessing
import numpy as np
from agent import sensor
import env.flat_world
import matplotlib.pyplot as plt
import importlib
import time

class MPEnv(object):
    def __init__(self, env_spec, comp_agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        WorldClass = getattr(importlib.import_module("env.flat_evade_world"), env_spec["world"]["type"])
        self.world = WorldClass(env_spec["world"]["spec"])
        self.env_spec = env_spec
        self.comp_agents = comp_agents
        self.reset()

    def reset(self):
        self.world.reset()
        for i in range(len(self.comp_agents)):
            self.world.add_agent(
                self.comp_agents[i],
                self.env_spec['agent_env_spec'][self.comp_agents[i].name])
        
        env_info, sensor_data = self.world.measure()
        sensor_data['time'] = time.time()
        return env_info, sensor_data

    def step(self, mgr_actions, mgr_sensor_data, mgr_record, lock, iters, render=True):
        for i in range(iters):
            print(f"env {i}")
            actions = {}
            with lock:
                actions.update(mgr_actions)
            self.world.simulate(actions, actions[self.comp_agents[0].name]['dt'])
            env_info, sensor_data = self.world.measure()
            sensor_data['time'] = time.time()
            mgr_record.put((env_info, sensor_data))
            with lock:
                mgr_sensor_data.update(sensor_data)
            if render:
                self.render()

    def render(self):
        plt.cla()
        plt.axis([0, 100, 0, 100])
        x = []
        y = []
        for name, agent in self.world.agents.items():
            if 'goal' not in name:
                x.append(agent.pos[0])
                y.append(agent.pos[1])

        for name, agent in self.world.agents.items():
            if 'goal' in name:
                x.append(agent.pos[0])
                y.append(agent.pos[1])
        
        cs = ['#ff0000', '#0000ff', '#ff5500', '#3399ff']
        plt.scatter(x,y,s=100, color=cs[:len(x)])
        # plt.plot(human_traj[:,0],human_traj[:,1])
        # plt.plot(robot_traj[:,0],robot_traj[:,1])
        plt.draw()
        plt.pause(0.001)
