import numpy as np
import evaluator, agent, env
import time
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"FlatReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.03, "disc_flag":1}},
        "estimator": {"type":"UKFEstimator",    "spec":{"init_x":np.array([ 10, 10, 0, 0]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":3,"kv":4}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"kp":6,"kv":8}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",      "spec":{"alias":"goal_sensor",     "noise_var":0.0}}],
    }

    agent_specs = [agent1_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"BB8Agent", "spec":{"init_x":np.vstack([ 0., 0., 0., 0.])}},}
    reaching_world_spec = {
        "friction": 0,
        "reaching_eps": 1,
        "agent_goal_lists":{
            "robot": [[20,20]],
        }
    }
    env_spec = {
        "world": {"type":"ReachingWorld", "spec":reaching_world_spec},
        "dt": 0.03,
        "agent_env_spec": agent_env_spec
    }
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.ModelBasedAgent(evaluator.agent_specs[i]))

    env = env.FlatEnv(env_spec, agents)
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
