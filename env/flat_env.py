from signal import pause
import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib
import time, math
import copy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image




class FlatEnv(object):
    def __init__(self, env_spec, comp_agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''

        self.env_spec = copy.deepcopy(env_spec)
        self.dt = self.env_spec['dt']
        WorldClass = getattr(importlib.import_module("env.flat_world"), self.env_spec["world"]["type"])
        self.world = WorldClass(self.env_spec["world"]["spec"])
        self.comp_agents = comp_agents
        self.reset()

        # setup rendering
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        # self.renderer = self.fig.canvas.renderer
        self.canvas = FigureCanvas(self.fig)

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
        img = None
        if render:
            img = self.render()
        return self.dt, env_info, sensor_data, img

    def render(self):
        
        self.ax.cla()

        self.ax.axis('equal')
        self.ax.set(xlim=(0, 101), ylim=(0, 101))

        # obs location
        c_obs = '#A2AEAF'
        for name, agent in self.world.agents.items():
            if 'obs' in name and 'goal' not in name:
                circ = plt.Circle(
                    agent.pos, 3.0, color='k', clip_on=False,
                    fill=False)
                self.ax.add_patch(circ)
                # self.ax.scatter(agent.pos[0],agent.pos[1],s=100, color='k')

        # agents location
        cs_agent = ['#ff0000', '#0000ff']
        x_agent = []
        y_agent = []
        for name, agent in self.world.agents.items():
            if 'goal' not in name and 'obs' not in name:
                x_agent.append(agent.pos[0])
                y_agent.append(agent.pos[1])
        
        self.ax.scatter(x_agent,y_agent,s=100, color=cs_agent[:len(x_agent)])

        # goal location
        cs_goal = ['#ff5500', '#3399ff']
        x_goal = []
        y_goal = []
        for name, agent in self.world.agents.items():
            if 'goal' in name and 'obs' not in name:
                x_goal.append(agent.pos[0])
                y_goal.append(agent.pos[1])
        
        self.ax.scatter(x_goal,y_goal,s=100, color=cs_goal[:len(x_goal)])
        
        # agent heading    return args[0]._bind(args[1:], kwargs)    return args[0]._bind(args[1:], kwargs)
        r = 3.0
        for name, agent in self.world.agents.items():
            if 'goal' not in name and agent.has_heading:
                xc, yc = agent.pos.reshape(-1)
                t = agent.heading
                self.ax.plot([xc, xc+r*math.cos(t)], [yc, yc+r*math.sin(t)],
                    color='b', linestyle='-')

        # self.ax.plot(human_traj[:,0],human_traj[:,1])
        # self.ax.plot(robot_traj[:,0],robot_traj[:,1])

        # self.ax.draw(self.renderer)
        # self.fig.canvas.draw()
        self.canvas.draw()       # draw the canvas, cache the renderer
        img_buf = io.BytesIO()
        self.fig.savefig(img_buf, format='png')
        plt.pause(0.001)

        return img_buf
    
    def close(self):
        plt.close(self.fig)

