import importlib
import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import evaluator, agent, env
import yaml
import multiprocessing

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
        agents.append(agent.MPWrapper(class_by_name('agent', type)(spec)))
    environ = env.MPWrapper(class_by_name('env', env_type)(env_spec, agents))
    evaluator = evaluator.Evaluator(agent_specs[0], env_spec)

    manager = multiprocessing.Manager()

    # initialize shared memory for record, sensor data, and actions
    mgr_record = manager.Queue()
    mgr_sensor_data = manager.dict()
    dt, init_env_info, init_sensor_data = environ.reset()
    mgr_record.put((init_env_info, init_sensor_data))
    mgr_sensor_data.update(init_sensor_data)
    lock = manager.Lock()

    mgr_actions = manager.dict()
    for ag in agents:
        init_actions = ag.init_action(init_sensor_data)
        with lock:
            mgr_actions[ag.name] = init_actions

    iters = config_spec['iters']
    mgr_running = manager.Value('b', True)

    # agent and env processes
    env_process = multiprocessing.Process(target=environ.step_loop, args=(mgr_actions, mgr_sensor_data, mgr_record, mgr_running, lock, iters))
    agent_processes = [multiprocessing.Process(target=ag.action_loop, args=(mgr_actions, mgr_sensor_data, mgr_running, lock)) for ag in agents]

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
