import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib
import time, math
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
        # add world agent for all computational agents
        for i in range(len(self.comp_agents)):
            self.world.add_agent(self.comp_agents[i],
                self.env_spec['agent_env_spec'][self.comp_agents[i].name])
        # add world agent for all non-computational agents

        env_info, sensor_data = self.world.measure()
        return self.dt, env_info, sensor_data

    def step(self, actions, render=True):
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
        if render:
            self.render()
        return self.dt, env_info, sensor_data

    def render(self):
        plt.cla()

        # obs location
        c_obs = '#A2AEAF'
        for name, agent in self.world.agents.items():
            if 'obs' in name:
                circ = plt.Circle(agent.pos, 5.0, color=c_obs, clip_on=False)
                ax = plt.gca()
                ax.add_patch(circ)

        # agents location
        cs = ['#ff0000', '#0000ff', '#ff5500', '#3399ff']
        x = []
        y = []
        for name, agent in self.world.agents.items():
            if 'goal' not in name and 'obs' not in name:
                x.append(agent.pos[0])
                y.append(agent.pos[1])

        for name, agent in self.world.agents.items():
            if 'goal' in name and 'obs' not in name:
                x.append(agent.pos[0])
                y.append(agent.pos[1])
        
        plt.scatter(x,y,s=100, color=cs[:len(x)])
        
        # agent heading    return args[0]._bind(args[1:], kwargs)    return args[0]._bind(args[1:], kwargs)
        r = 3.0
        for name, agent in self.world.agents.items():
            if 'goal' not in name and agent.has_heading:
                xc, yc, t = agent.pos.reshape(-1)
                plt.plot([xc, xc+r*math.cos(t)], [yc, yc+r*math.sin(t)],
                    color='b', linestyle='-')

        # plt.plot(human_traj[:,0],human_traj[:,1])
        # plt.plot(robot_traj[:,0],robot_traj[:,1])

        # plt.axis('equal')
        plt.gca().set_xlim((0, 100))
        plt.gca().set_ylim((0, 100))
        plt.draw()
        plt.pause(0.001)
