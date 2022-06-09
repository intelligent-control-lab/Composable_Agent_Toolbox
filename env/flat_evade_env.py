import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib
import time
class FlatEvadeEnv(object):
    def __init__(self, env_spec, comp_agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''
        self.dt = env_spec['dt']
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
        return self.dt, env_info, sensor_data

    def step(self, actions, debug_modes, render=True):
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
        if render:
            self.render(actions, debug_modes)
        return self.dt, env_info, sensor_data

    def render(self, actions, debug_modes):
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

        if debug_modes['render_traj']:
            traj_x = []
            traj_y = []
            for pt in actions[self.comp_agents[0].name]['broadcast']['planned_traj']:
                traj_x.append(pt[0])
                traj_y.append(pt[1])
            plt.plot(traj_x, traj_y, color='black')
        if debug_modes['render_next_traj_point']:
            next_point = actions[self.comp_agents[0].name]['broadcast']['next_point']
            plt.scatter([next_point[0]], [next_point[1]], c='gray')
        
        cs = ['#ff0000', '#0000ff', '#ff5500', '#3399ff']
        plt.scatter(x,y,s=100, color=cs[:len(x)])
        # plt.plot(human_traj[:,0],human_traj[:,1])
        # plt.plot(robot_traj[:,0],robot_traj[:,1])
        plt.plot()
        plt.draw()
        plt.pause(0.001)
