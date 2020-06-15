import numpy as np
import evaluator, agent, environment

if __name__ == "__main__":

    module_spec = {
        "task": {(1,1):10, (2,2):20}, # we define task as a state-goal mapping here, but better design is needed
        "model": "Model",
        "estimator": "Estimator",
        "planner": "Planner",
        "controller": {"feedback": "PID", "params": {"kp": [1, 1], "ki": [0, 0], "kd": [0, 0]}},
        "sensors": ["PVSensor", "PVSensor"], #an agent can have multiple sensors
    }
    agent_specs = [module_spec, module_spec]
    agent_env_spec = [{'init_x':np.vstack([0,0,0,0])}, 
                      {'init_x':np.vstack([0,0,0,0])}]
    env_spec = {
        "dt": 0.1,
        "friction": 0,
        "agent_env_spec": agent_env_spec
    }
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.Agent(evaluator.agent_specs[i]))

    env = environment.Environment(env_spec, agents)
    dt, env_info, agent_sensor_data = env.reset()
    record = []

    
    for it in range(100):
        print("iter = ",it)

        actions = []
        for i in range(len(agents)):
            actions.append(agents[i].action(dt, agent_sensor_data[i]))
            #sensor data is grouped by agent
        dt, env_info, agent_sensor_data = env.step(actions)
        record.append(env_info)

    evaluator.evaluate(record)

