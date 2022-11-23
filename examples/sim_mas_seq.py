import importlib
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
        ax.scatter(x, y, t, color=colors[i])
    plt.show()
    return

if __name__ == '__main__':

    # config.yaml would contain yaml filenames of user's desired agents/env
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
    r = 0.35 # TODO: perhaps allow for different paths to have different r?
    ag_dt = np.array([ag.dt for ag in agents])
    a_max = [10 for ag in agents]
    gamma = [1 for ag in agents]
    priority = [1 for ag in agents]
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

    # # plot obstacle paths
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # colors = 'brgcmk'
    # for i, p in enumerate(stg.obs_paths):
    #     x = [s[0] for s in p]
    #     y = [s[1] for s in p]
    #     t = [s[2] for s in p]
    #     ax.scatter(x, y, t, color=colors[i])
    # plt.show()

    print("Simulation progress:")
    for it in progressbar.progressbar(range(iters)):

        # print("Iterating...")
        
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
        
    visualize(stg)

    print(f"OPT NUM: {stg.opt_num}")
    print(f"TREE TIME: {stg.tree_time}")
    print(f"AVG OPTS PER RES {stg.opt_num / stg.res_num}")
    print(f"RESOLVE TIME: {stg.resolve_time - stg.tree_time}")

    makespan = stg.paths[0][-1][2]
    for i in range(1, len(stg.paths)):
        makespan = max(makespan, stg.paths[i][-1][2])
    print(f"MAKESPAN: {makespan}")

    tot_dist = 0
    for p in stg.paths:
        for i in range(len(p) - 1):
            tot_dist += np.linalg.norm(p[i + 1] - p[i])
    print(f"TOTAL DISTANCE: {tot_dist}")

    min_clear = np.inf
    for i, p1 in enumerate(stg.paths + stg.obs_paths):
        for p2 in (stg.paths + stg.obs_paths)[i + 1:]:
            for s1 in p1:
                for s2 in p2:
                    min_clear = min(min_clear, np.linalg.norm(s1 - s2) / stg.alpha)
    print(f"MIN CLEARANCE FINAL: {min_clear}")

    dt, env_info, measurement_groups = env.reset()
    for i in range(iters):
        actions = {}
        for ag in agents + obs:
            # an action is dictionary which must contain a key "control"
            actions[ag.name] = ag.action(dt, measurement_groups[ag.name])
            #sensor data is grouped by agent
            print(f"{ag.name} : {actions[ag.name]['control']} : {actions[ag.name]['broadcast']['next_point']}")
        dt, env_info, measurement_groups = env.step(actions, debug_modes, render=render)

    # evaluator.evaluate(record)