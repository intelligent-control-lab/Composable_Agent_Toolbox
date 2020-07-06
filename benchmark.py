import numpy as np
import evaluator, agent, environment
import time
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"ReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{}},
        "planner":   {"type":"OptimizationBasedPlanner",    "spec":{"horizon":10, "replanning_cycle":10, "dim":2, "n_ob":0}},
        "controller":{"type":"NaiveController", "spec":{"speed_factor":10}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",      "spec":{"alias":"goal_sensor",     "noise_var":0.0}}],
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"ReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{}},
        "planner":   {"type":"OptimizationBasedPlanner",    "spec":{"horizon":20, "replanning_cycle":10, "dim":2, "n_ob":0}},
        "controller":{"type":"NaiveController", "spec":{"speed_factor":10}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",      "spec":{"alias":"goal_sensor",     "noise_var":0.0}}],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"BB8Agent", "init_x":np.vstack([ 0, 0, 0, 0])},
                      "human":{"type":"BB8Agent", "init_x":np.vstack([ 0,10, 0, 0])}
                    }
    reaching_world_spec = {
        "friction": 0,
        "reaching_eps": 1,
        "agent_goal_lists":{
            "robot": [[10,20],[20,10],[30,20],[40,10]],
            "human": [[20,30],[20,20],[40,30],[40,20]],
        }
    }
    env_spec = {
        "world": {"type":"ReachingWorld", "spec":reaching_world_spec},
        "dt": 0.1,
        "agent_env_spec": agent_env_spec
    }
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.Agent(evaluator.agent_specs[i]))

    env = environment.Environment(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()
    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(1000)):
        actions = {}
        for agent in agents:
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append(env_info)

    evaluator.evaluate(record)
