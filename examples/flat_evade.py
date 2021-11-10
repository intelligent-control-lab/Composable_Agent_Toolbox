import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time, yaml
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    with open('configs/flat_evade_agent_1.yaml', 'r') as infile:
        agent1_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    
    with open('configs/flat_evade_agent_2.yaml', 'r') as infile:
        agent2_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents
    agents = []
    for i in range(len(agent_specs)):
        agents.append(agent.ModelBasedAgent(agent_specs[i]))

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('configs/flat_evade_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    env = env.FlatEvadeEnv(env_spec, agents)
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