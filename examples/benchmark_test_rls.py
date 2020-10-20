import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time
# import progressbar

if __name__ == "__main__":

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    agent1_module_spec = {
        "name":       "robot",
        "task":      {"type":"ReachingTask",            "spec":{}},
        "model":     {"type":"LinearModel",             "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.02, "disc_flag":1}},
        "estimator": {"type":"RLSPredictor",            "spec":{"init_x":np.array([ 30.,20.0, 0., 0.]),"other_goal":np.array([10.0,20.0,0.0,0.0]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"OptimizationBasedPlanner","spec":{"horizon":10, "replanning_cycle":10, "dim":2, "n_ob":1, "obs_r":5}},
        "controller":{"type":"CBFController",           "spec":{"kp":6,"kv":8}},
        "sensors":  [{"type":"PVSensor",                "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",             "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",             "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",              "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",             "spec":{"alias":"communication_sensor"}},
        ],
    }

    agent2_module_spec = {
        "name":       "human",
        "task":      {"type":"ReachingTask",    "spec":{}},
        "model":     {"type":"LinearModel",     "spec":{"use_library":0, "model_name":'Ballbot', "time_sample":0.02, "disc_flag":1}},
        "estimator": {"type":"RLSPredictor",  "spec":{"init_x":np.array([50.,20.0, 0., 0.]),"other_goal":np.array([70.0,20.0,0.0,0.0]),"init_variance":.01*np.eye(4),"Rww":.001*np.eye(4),"Rvv":.001*np.eye(4),"alpha_ukf":1,"kappa_ukf":0.1,"beta_ukf":2,"time_sample":0.01,"kp":6,"kv":8}},
        "planner":   {"type":"NaivePlanner",    "spec":{"horizon":20, "replanning_cycle":10}},
        "controller":{"type":"CBFController", "spec":{"kp":6,"kv":8}},
        "sensors":  [{"type":"PVSensor",        "spec":{"alias":"cartesian_sensor","noise_var":0.1}},
                     {"type":"StateSensor",     "spec":{"alias":"state_sensor",    "noise_var":0.1}},
                     {"type":"RadarSensor",     "spec":{"alias":"obstacle_sensor", "noise_var":0.1}}, #an agent can have multiple sensors
                     {"type":"GoalSensor",      "spec":{"alias":"goal_sensor",     "noise_var":0.0}},
                     {"type":"RadioSensor",     "spec":{"alias":"communication_sensor"}}],
    }

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    agent_env_spec = {"robot":{"type":"BB8Agent", "init_x":np.vstack([ 30.,20.0, 0., 0.])},
                      "human":{"type":"BB8Agent", "init_x":np.vstack([ 50.,20.0, 0., 0.])}
                    }
    reaching_world_spec = {
        "friction": 0,
        "reaching_eps": 0.1,
        "agent_goal_lists":{
            "robot": [[70.0,20.0]],
            "human": [[10.0,20.0]],
        }
    }
    env_spec = {
        "world": {"type":"ReachingWorld", "spec":reaching_world_spec},
        "dt": 0.02,
        "agent_env_spec": agent_env_spec
    }
    evaluator = evaluator.Evaluator(agent_specs, env_spec)

    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.ModelBasedAgent(evaluator.agent_specs[i]))

    env = env.FlatEnv(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()
    record = []
    human_traj = []
    robot_traj = []
    print("Simulation progress:")
    for it in (range(1000)):
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            if agent.name == "robot":
                for agent2 in agents:
                    if agent2.name == "human":
                        human_state = agent2.get_state(dt, measurement_groups[agent2.name])
                actions[agent.name], robot_traj = agent.action(dt, measurement_groups[agent.name], human_traj, human_state)
            if agent.name == "human":
                for agent2 in agents:
                    if agent2.name == "robot":
                        robot_state = agent2.get_state(dt, measurement_groups[agent2.name])
                actions[agent.name], human_traj = agent.action(dt, measurement_groups[agent.name], robot_traj, robot_state)
            # print(f"name: {agent.name}, control: {actions[agent.name]['control']}")
            #sensor data is grouped by agent
        dt, env_info, measurement_groups = env.step(actions, human_traj, robot_traj)
        record.append(env_info)

    evaluator.evaluate(record)