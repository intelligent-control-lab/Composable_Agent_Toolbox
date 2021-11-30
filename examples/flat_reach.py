import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time, yaml
import progressbar

if __name__ == "__main__":

    agent_specs = []

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('configs/flat_reach_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    # create computational agents
    agents = []
    for agent_name, agent_spec_file in env_spec['agent_comp_spec'].items():
        with open(agent_spec_file, 'r') as infile:
            agent_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
            agents.append(agent.ModelBasedAgent(agent_module_spec))

    # init env
    env = env.FlatEnv(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()
    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(500)):
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append((env_info,measurement_groups))

    evaluator.evaluate(record)