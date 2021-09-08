import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time
import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"FlatReachingTask",        "spec":{}},
        "model":     {"type":"LinearModel",             "spec":{"use_spec":0, "use_library":0, "model_name":'Ballbot', "time_sample":0.02, "disc_flag":1, "model_spec":None,"control_input_dim":2}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([50.,20.0, 0., 0.]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"OptimizationBasedPlanner",    "spec":{"horizon":20, "replanning_cycle":10, "dim":2, "n_ob":1, "obs_r":5}},
        "controller":{"type":"NaiveController",         "spec":{"kp":6,"kv":8}},
        "sensors":  [{"type":"PVSensor",                "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}},
        ],
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"FlatReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_spec":0, "use_library":0, "model_name":'Ballbot', "time_sample":0.02, "disc_flag":1, "model_spec":None,"control_input_dim":2}},
        "estimator": {"type":"NaiveEstimator",  "spec":{"init_x":np.array([50.,20.0, 0., 0.]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"OptimizationBasedPlanner",    "spec":{"horizon":20, "replanning_cycle":10, "dim":2, "n_ob":1, "obs_r":5}},
        "controller":{"type":"NaiveController", "spec":{"kp":6,"kv":8}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",      "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",     "spec":{"alias":"communication_sensor"}}],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents
    agents = []
    for i in range(len(agent_specs)):
        agents.append(agent.ModelBasedAgent(agent_specs[i]))

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"BB8Agent", "spec":{"init_x":np.vstack([ 30.,20.0, 0., 0.])}},
                      "human":{"type":"BB8Agent", "spec":{"init_x":np.vstack([ 50.,20.0, 0., 0.])}}
                    }
    reaching_world_spec = {
        "friction": 0,
        "reaching_eps": 0.1,
        "agent_goal_lists":{
            "robot": [[70.0,20.0], [10, 40]],
            "human": [[10.0,20.0], [40, 70]],
        }
    }
    env_spec = {
        "world": {"type":"FlatReachingWorld", "spec":reaching_world_spec},
        "dt": 0.02,
        "agent_env_spec": agent_env_spec
    }
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    env = env.FlatEnv(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()
    record = []
    print("Simulation progress:")
    # for it in progressbar.progressbar(range(100)):
    for it in progressbar.progressbar(range(100)):
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions)
        record.append((env_info,measurement_groups))

    evaluator.evaluate(record)