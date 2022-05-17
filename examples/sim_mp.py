import multiprocessing
import yaml

import agent
import env
import evaluator

if __name__ == '__main__':

    # config.yaml would contain yaml filenames of user's desired agents/env
    with open('examples/configs/config.yaml', 'r') as infile:
        config_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    env_spec_file = config_spec['env']
    with open(env_spec_file, 'r') as infile:
        env_spec = yaml.load(env_spec_file, Loader=yaml.SafeLoader)

    agent_specs = []
    for agent_name, agent_spec_file in config_spec['agents'].items():
        with open(agent_spec_file, 'r') as infile:
            agent_spec = yaml.load(infile, Loader=yaml.SafeLoader)
            agent_specs.append(agent_spec)

    # TODO: figure out how to customize TYPE of agent and TYPE of env (e.g. modelfree, flatevade)
    agents = [agent.MPWrapper(agent.ModelBasedAgent(spec)) for spec in agent_specs]
    environ = env.MPWrapper(env.FlatEnv(env_spec, agents))
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

    # TODO: make iter num configurable?
    iters = 200
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
