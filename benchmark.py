import numpy as np
import evaluator, agent, environment

if __name__ == "__main__":

    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"ReachingTask",    "spec":{"reaching_eps":1, "goal_range":[[10,10,0,0],[90,90,0,0]]}},
        "model":     {"type":"Model",           "spec":{}},
        "estimator": {"type":"NaiveEstimator",  "spec":{}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"speed_factor":10}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}], #an agent can have multiple sensors
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"ReachingTask",    "spec":{"reaching_eps":1, "goal_range":[[10,10,0,0],[90,90,0,0]]}},
        "model":     {"type":"Model",           "spec":{}},
        "estimator": {"type":"NaiveEstimator",  "spec":{}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"speed_factor":10}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}], #an agent can have multiple sensors
    }

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

 