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

        # setup rendering
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        # self.renderer = self.fig.canvas.renderer

    def reset(self):
        self.world.reset()
        # add world agent for all computational agents
        for i in range(len(self.comp_agents)):
            self.world.add_agent(self.comp_agents[i],
                self.env_spec['agent_env_spec'][self.comp_agents[i].name])
        # add world agent for all non-computational agents

        env_info, sensor_data = self.world.measure()
        return self.dt, env_info, sensor_data

    def step(self, actions, debug_modes, render=True):
        self.world.simulate(actions, self.dt)
        env_info, sensor_data = self.world.measure()
        if render:
            self.render(actions, debug_modes)
        self.log_debug(sensor_data, debug_modes)
        return self.dt, env_info, sensor_data

    def log_debug(self, sensor_data, debug_modes):
        if debug_modes['log_agent_state']:
            for ag in self.comp_agents:
                state = sensor_data[ag.name]['cartesian_sensor']
                print(f"({ag.name}) pos: {state['pos']}\n")
                print(f"({ag.name}) vel: {state['vel']}\n")

    def render(self, actions, debug_modes):
        
        self.ax.cla()

        self.ax.axis('equal')
        self.ax.set(xlim=(0, 101), ylim=(0, 101))

        # obs location
        c_obs = '#A2AEAF'
        for name, agent in self.world.agents.items():
            if 'obs' in name:
                circ = plt.Circle(
                    agent.pos, 5.0, color='k', clip_on=False,
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

        if debug_modes['render_traj']:
            traj_x = []
            traj_y = []
            for pt in actions[self.comp_agents[0].name]['broadcast']['planned_traj']:
                traj_x.append(pt[0][0])
                traj_y.append(pt[0][1])
            self.ax.plot(traj_x, traj_y, color='black')
        if debug_modes['render_next_traj_point']:
            next_point = actions[self.comp_agents[0].name]['broadcast']['next_point']
            self.ax.scatter([next_point[0]], [next_point[1]], c='gray')

        # self.ax.plot(human_traj[:,0],human_traj[:,1])
        # self.ax.plot(robot_traj[:,0],robot_traj[:,1])

        # self.ax.draw(self.renderer)
        self.fig.canvas.draw()
        plt.pause(0.001)
