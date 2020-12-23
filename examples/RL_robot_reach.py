import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time
import torch
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    # This is only for demonstration, the RL module is not well trained.
    agent1_module_spec = {
        "name":         "robot",
        "policy":       {"type":"ActorCriticPolicy", "spec":{"model_path": join(abspath(dirname(__file__)), '../agent/saved_models/FetchReach-v1/model.pt'),
                                                             "env_params": {
                                                                            "obs":          10, 
                                                                            "goal":         3, 
                                                                            "action":       4, 
                                                                            "action_max":   torch.tensor([1., 1., 1., 1.], dtype=torch.float32),
                                                                            },
                                                            }
                        },
        "sensors":      [{"type":"PVSensor",               "spec":{"alias":"cartesian_sensor","noise_var":0.0}},
                        {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.0}},
                        {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.0}}, #an agent can have multiple sensors
                        {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                        {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}}],
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"FrankaReachingTask",      "spec":{}},
        "model":     {"type":"LinearModel",             "spec":{"use_spec":0, "use_library":0, "model_name":'Ballbot', "time_sample":0.01, "disc_flag":1, "model_spec":None,"control_input_dim":7}},
        "estimator": {"type":"NaiveEstimator",            "spec":{"name":"human","init_x":np.array([1.1,0.0,0.0]),"init_variance":.01*np.eye(3),"Rww":.001*np.eye(3),"Rvv":.001*np.eye(3),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.1,"kp":1,"kv":3}},
        "planner":   {"type":"NaivePlanner",            "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"NaiveController",         "spec":{"kp":0.5,"kv":0.1}},
        "sensors":  [{"type":"PVSensor",                "spec":{"alias":"cartesian_sensor","noise_var":0.0}},
                     {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.0}},
                     {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.0}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}}],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents
    agents = [agent.ModelFreeAgent(agent1_module_spec), agent.ModelBasedAgent(agent2_module_spec)]

    # agent_specs = [agent1_module_spec] # specs for two agents
    # agents = [agent.ModelFreeAgent(agent1_module_spec)]

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"FrankaPanda","spec":{"control_space":"cartesian",  "base_position":[0,0,0],   "init_joints":[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]}},
                      "human":{"type":"Ball",       "spec":{"init_position":[1.1,0,0.2]}},
                    #   "human":{"type":"ManualBall",       "spec":{"init_position":[1.1,0,0]}}
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

    env = env.BulletEnv(env_spec, agents)

    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    dt, env_info, measurement_groups = env.reset()
    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(100)):
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append((env_info,measurement_groups))

    evaluator.evaluate(record)