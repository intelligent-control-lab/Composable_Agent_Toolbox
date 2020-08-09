import numpy as np
import evaluator, agent, env
import time
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"FrankaReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([50.,20.0, 0., 0.]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"kp":2,"kv":0.1}},
        # "sensors":  []
        "sensors":  [{"type":"PVSensor",                "spec":{"alias":"cartesian_sensor","noise_var":0.0}},
                     {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.0}},
                     {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.0}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}}],
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"FrankaReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([50.,20.0, 0., 0.]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController", "spec":{"kp":0.5,"kv":0.1}},
        "sensors":  [],
        "sensors":  [{"type":"PVSensor",                "spec":{"alias":"cartesian_sensor","noise_var":0.0}},
                     {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.0}},
                     {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.0}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}}],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"FrankaPanda","spec":{"base_position":[0,0,0],   "init_joints":[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]}},
                      "human":{"type":"Ball",       "spec":{"init_position":[1.1,0,0],}}
                    }

    # agent_specs = []
    # agent_env_spec = {}

    bullet_world_spec = {
        "gravity": 10,
        "reaching_eps": 0.05,
        "agent_goal_lists":{
            "robot": [[0.5, -0.3, 0.2], [0.5, -0.3, 0.4], [0.5, -0.1, 0.2]],
            "human": [[0.5, 0.1, 0.5], [0.6, 0.2, 0.2], [0.4, -0.3, 0.5], [0.6, -0.3, 0.2], ],
        }
    }
    env_spec = {
        "world": {"type":"BulletWorld", "spec":bullet_world_spec},
        "dt": 0.05,
        "agent_env_spec": agent_env_spec,
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
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
        
        dt, env_info, measurement_groups = env.step(actions)
        record.append(env_info)
        