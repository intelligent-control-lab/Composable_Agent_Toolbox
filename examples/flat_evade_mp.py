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

    with open('configs/flat_evade_agent_1_mp.yaml', 'r') as infile:
        agent_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    agent1 = agent.ModelBasedAgentMP(agent_module_spec)
    agents = [agent1]

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('configs/flat_evade_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    evaluator = evaluator.Evaluator(agent_module_spec, env_spec)
    env = env.FlatEvadeEnvMP(env_spec, agents)

    manager = multiprocessing.Manager()

    # initialize shared memory for record, sensor data, and actions
    mgr_record = manager.Queue()
    mgr_sensor_data = manager.dict()
    init_env_info, init_sensor_data = env.reset()
    mgr_record.put((init_env_info, init_sensor_data))
    mgr_sensor_data.update(init_sensor_data)

    mgr_actions = manager.dict()
    init_actions = agent1.init_action(init_sensor_data)
    mgr_actions[agent1.name] = init_actions

    lock = manager.Lock()
    iters = 500

    # agent and env processes
    agent_process = multiprocessing.Process(target=agent1.action, args=(mgr_actions, mgr_sensor_data, lock, iters))
    env_process = multiprocessing.Process(target=env.step, args=(mgr_actions, mgr_sensor_data, mgr_record, lock, iters))

    agent_process.start()
    env_process.start()

    agent_process.join()
    env_process.join()
    
    # collect and evaluate record data
    record = []
    while not mgr_record.empty():
        record.append(mgr_record.get())
    evaluator.evaluate(record)
