import numpy as np
import evaluator, agent, env
import time
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot1",
        "task":      {"type":"ReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([ 10, 10, 0, 0]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"time_sample":0.01,"kp":3,"kv":4}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"kp":3,"kv":4}},
        "sensors":  [],
    }

    agent2_module_spec = {
        "name":       "robot2",
        "task":      {"type":"ReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([ 10, 10, 0, 0]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"time_sample":0.01,"kp":3,"kv":4}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"kp":3,"kv":4}},
        "sensors":  [],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot1":{"type":"FrankaPanda",  "spec":{"base_position":[0,0,0],   "init_joints":[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]}},
                      "robot2":{"type":"FrankaPanda",  "spec":{"base_position":[1.1,0,0], "init_joints":[3.14,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]}}
                    }

    # agent_specs = []
    # agent_env_spec = {}

    bullet_world_spec = {
        "gravity": 10,
        "reaching_eps": 1,
    }
    env_spec = {
        "world": {"type":"BulletWorld", "spec":bullet_world_spec},
        "dt": 0.05,
        "agent_env_spec": agent_env_spec
    }

    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.ModelBasedAgent(evaluator.agent_specs[i]))

    env = env.BulletEnv(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()

    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(1000)):
        actions = {}
        for agent in agents:
            # actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            actions[agent.name] = {
                "control": [1,1,1,1]
            }
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append(env_info)
        
