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
    for i, p in enumerate(stg.paths):
        x = [s[0] for s in p]
        y = [s[1] for s in p]
        t = [s[2] for s in p]
        ax.scatter(x, y, t, color=colors[i])
    plt.show()

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

    agents = []
    for type, spec in zip(agent_types, agent_specs):
        agents.append(class_by_name('agent', type)(spec))
    # env = class_by_name('env', env_type)(env_spec, agents)
    # evaluator = evaluator.Evaluator(agent_specs[0], env_spec)

    iters = config_spec['iters']
    debug_modes = {mode: val for mode, val in config_spec['debug'].items()}
    render = config_spec['render']

    # dt, env_info, measurement_groups = env.reset()
    # record = []
    dt = np.array([ag.dt for ag in agents])
    stg = SpaceTimeGrid([[np.array(ag.path[0][:2])] for ag in agents], np.ones(len(agents)), np.ones(len(agents)), np.ones(len(agents)), dt)
    stg.vel = [[np.array(ag.path[0][2:])] for ag in agents]

    print("Simulation progress:")
    for it in progressbar.progressbar(range(iters)):

        print("Iterating...")
        
        for i, agent in enumerate(agents):
            if agent.at_goal:
                stg.set_at_goal(i, True)
                continue
            # an action is dictionary which must contain a key "control"
            waypoint = agent.next_point()
            stg.update_path(i, waypoint[0], waypoint[1])
            stg.resolve()
            for j, a in enumerate(agents):
                path = stg.get_path(j)
                a.set_path(path)
            #sensor data is grouped by agent
        # dt, env_info, measurement_groups = env.step(actions, debug_modes, render=render)
        # record.append((env_info,measurement_groups))
        
    visualize(stg)

    print(stg.paths[0])
    print('\n')
    print(stg.paths[1])

    # evaluator.evaluate(record)