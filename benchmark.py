import numpy as np
import evaluator, agent, environment

if __name__ == "__main__":

    agent_module_spec = {
        "task": {(1,1):10, (2,2):20}, # we define task as a state-goal mapping here, but better design is needed
        "model": "Model",
        "estimator": "Estimator",
        "planner": "Planner",
        "controller": {"feedback": "PID", "params": {"kp": [1, 1], "ki": [0, 0], "kd": [0, 0]}},
        "sensors": [{"type":"PVSensor", "alias":"state_sensor", "noise_var":0.1}, 
                    {"type":"RadarSensor", "alias":"obstacle_sensor", "noise_var":0.1}], #an agent can have multiple sensors
    }
    agent1_module_spec = {**agent_module_spec, **{"name":"robot"}}
    agent2_module_spec = {**agent_module_spec, **{"name":"human"}}
    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents
    agent_env_spec = {"robot":{"type":"BB8Agent", "init_x":np.vstack([ 0, 0, 0, 0])},
                      "human":{"type":"BB8Agent", "init_x":np.vstack([ 0,10, 0, 0])}
                    }
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
    dt, env_info, measurement_groups = env.reset()
    record = []
    
    for it in range(100):
        print("iter = ",it)
        actions = []
        for i in range(len(agents)):
            actions.append(agents[i].action(dt, measurement_groups[agents[i].name]))
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append(env_info)

    evaluator.evaluate(record)

 