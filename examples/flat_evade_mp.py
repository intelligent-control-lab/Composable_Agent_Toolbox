import sys
from os.path import abspath, join, dirname

from sympy import init_session
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time, yaml
import progressbar
import multiprocessing

if __name__ == "__main__":

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('configs/flat_evade_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    agent_specs = []
    for agent_name, agent_spec_file in env_spec['agent_comp_spec'].items():
        with open(agent_spec_file, 'r') as infile:
            agent_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
            agent_specs.append(agent_module_spec)
    agents = [agent.ModelBasedAgentMP(spec) for spec in agent_specs]

    env = env.MPWrapper(env.FlatEvadeEnv(env_spec, agents))
    evaluator = evaluator.Evaluator(agent_specs[0], env_spec)

    manager = multiprocessing.Manager()

    # initialize shared memory for record, sensor data, and actions
    mgr_record = manager.Queue()
    mgr_sensor_data = manager.dict()
    dt, init_env_info, init_sensor_data = env.reset()
    mgr_record.put((init_env_info, init_sensor_data))
    mgr_sensor_data.update(init_sensor_data)

    mgr_actions = manager.dict()
    for agent in agents:
        init_actions = agent.init_action(init_sensor_data)
        mgr_actions[agent.name] = init_actions

    lock = manager.Lock()
    mgr_running = manager.Value('b', True)
    iters = 1000

    # agent and env processes
    env_process = multiprocessing.Process(target=env.step, args=(mgr_actions, mgr_sensor_data, mgr_record, mgr_running, lock, iters))
    agent_processes = [multiprocessing.Process(target=agent.action, args=(mgr_actions, mgr_sensor_data, mgr_running, lock)) for agent in agents]

    env_process.start()
    for proc in agent_processes:
        proc.start()

    env_process.join()
    for proc in agent_processes:
        proc.join()
    
    # collect and evaluate record data
    record = []
    while not mgr_record.empty():
        record.append(mgr_record.get())
    evaluator.evaluate(record)
