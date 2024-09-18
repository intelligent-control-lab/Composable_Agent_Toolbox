import importlib
import math
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import evaluator, agent, env
import yaml
import progressbar
from agent.multi_agent_system.SpaceTimeGrid import SpaceTimeGrid

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def class_by_name(module_name, class_name):
    return getattr(importlib.import_module(module_name), class_name)

def visualize(stg):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = 'brgcmk'
    if len(stg.obs_paths) == 1 and len(stg.obs_paths[0]) == 1:
        pat = stg.paths
    else:
        pat = stg.paths + stg.obs_paths
    for i, p in enumerate(pat):
        x = [s[0] for s in p]
        y = [s[1] for s in p]
        t = [s[2] for s in p]
        ax.scatter(x, y, t, s=400, color=colors[i])
    plt.show()
    return

def main():

    # config.yaml contains yaml filenames of user's desired agents/env
    with open('configs/config.yaml', 'r') as infile:
        config_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    env_spec_file = config_spec['env']['spec']
    env_type = config_spec['env']['type']
    with open(env_spec_file, 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    agent_types = []
    agent_specs = []
    for agent_name in config_spec['agents']:
        agent_types.append(config_spec['agents'][agent_name]['type'])
        spec_file = config_spec['agents'][agent_name]['spec']
        with open(spec_file, 'r') as infile:
            agent_spec = yaml.load(infile, Loader=yaml.SafeLoader)
        agent_specs.append(agent_spec)

    obs_types = []
    obs_specs = []
    for obs_name in config_spec['obs']:
        obs_types.append(config_spec['obs'][obs_name]['type'])
        spec_file = config_spec['obs'][obs_name]['spec']
        with open(spec_file, 'r') as infile:
            obs_spec = yaml.load(infile, Loader=yaml.SafeLoader)
        obs_specs.append(obs_spec)

    agents = []
    for type, spec in zip(agent_types, agent_specs):
        agents.append(class_by_name('agent', type)(spec))

    obs = []
    for type, spec in zip(obs_types, obs_specs):
        obs.append(class_by_name('agent', type)(spec))

    env = class_by_name('env', env_type)(env_spec, agents + obs)

    iters = config_spec['iters']
    debug_modes = {mode: val for mode, val in config_spec['debug'].items()}
    render = config_spec['render']

    ag_dt, env_info, measurement_groups = env.reset()
    paths = [[np.array(ag.path[0][:2])] for ag in agents]
    r = 0.35
    ag_dt = np.array([0.02 for ag in agents])
    a_max = [10 for ag in agents]
    gamma = [1 for ag in agents]
    priority = [1 for ag in agents]
    # priority[0] = 100
    obs_paths = [[np.array(ob.path[0][:2])] for ob in obs]
    obs_dt = np.array([ob.dt for ob in obs])
    stg = SpaceTimeGrid(paths, r, ag_dt, a_max, gamma, priority, obs_paths, obs_dt)

    for i, ob in enumerate(obs):
        while not ob.at_goal or ('stat' in ob.name and stg.obs_paths[i][-1][2] < 5):
            waypoint = ob.get_waypoint().flatten()
            stg.update_obs_path(i, waypoint[:2])
            path = ob.path.copy()
            path.append(waypoint)
            ob.set_path(path)

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    print("Simulation progress:")
    for it in progressbar.progressbar(range(iters)):
        for i, agent in enumerate(agents):
            if agent.at_goal:
                stg.set_at_goal(i, True)
                continue
            waypoint = agent.get_waypoint()
            stg.update_path(i, waypoint[0], waypoint[1])
            stg.resolve(it)
            for j, a in enumerate(agents):
                path = stg.get_path(j)
                a.set_path(path)

            # ax = fig.add_subplot(projection='3d')
            # colors = 'brgcmk'
            # if len(stg.obs_paths) == 1 and len(stg.obs_paths[0]) == 1:
            #     pat = stg.paths
            # else:
            #     pat = stg.paths + stg.obs_paths
            # for i, p in enumerate(pat):
            #     x = [s[0] for s in p]
            #     y = [s[1] for s in p]
            #     t = [s[2] for s in p]
            #     ax.axes.set_xlim3d(left=10, right=20)
            #     ax.axes.set_ylim3d(bottom=5, top=15)
            #     ax.axes.set_zlim3d(bottom=0, top=4)
            #     ax.scatter(x, y, t, s=400, color=colors[i])
            # plt.draw()
            # plt.pause(0.001)
            # ax.cla()

    visualize(stg)

if __name__ == '__main__':
    main()