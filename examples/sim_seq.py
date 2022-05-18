import importlib
import sys
from os.path import abspath, join, dirname

import progressbar

sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import evaluator, agent, env
import yaml

def class_by_name(module_name, class_name):
    return getattr(importlib.import_module(module_name), class_name)

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
    env = class_by_name('env', env_type)(env_spec, agents)
    evaluator = evaluator.Evaluator(agent_specs[0], env_spec)

    iters = config_spec['iters']

    dt, env_info, measurement_groups = env.reset()
    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(iters)):
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append((env_info,measurement_groups))

    evaluator.evaluate(record)