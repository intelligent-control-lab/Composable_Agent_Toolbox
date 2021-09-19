import sys, os
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time
import progressbar, yaml
from agent.planner.src.utils import forkine
from agent.planner.src.robot import RobotProperty
from math import pi

if __name__ == "__main__":

    # initial end effector position
    robot = RobotProperty()
    state_goal_list = np.array([[0,pi/2,0,0,0,pi,0], [0,pi/2,pi/4,-pi/4,0,0,0], [pi/4,1.5*pi/4,pi/4,-pi/4,0,pi/2,0]])
    cartesian_goal_list = [forkine(np.squeeze(g), robot.DH, robot.base)  for g in state_goal_list]

    # The module specs for agents, specifies which task, model, planner, controller, sensor to use.
    with open('examples/configs/model_based_robot_reach_agent_1.yaml', 'r') as infile:
        agent1_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    
    # fill goals
    agent1_module_spec['task']['spec']['cartesian_goal_list'] = cartesian_goal_list
    agent1_module_spec['task']['spec']['state_goal_list']     = state_goal_list

    with open('examples/configs/model_based_robot_reach_agent_2.yaml', 'r') as infile:
        agent2_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    agent_specs = [agent1_module_spec, agent2_module_spec] # specs for two agents

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('examples/configs/model_based_robot_reach_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)
    
    env_spec['world']['spec']['agent_goal_lists']['robot'] = cartesian_goal_list
    
    evaluator = evaluator.Evaluator(agent_specs, env_spec)
 
    agents = []
    for i in range(len(evaluator.agent_specs)):
        agents.append(agent.ModelBasedAgent(evaluator.agent_specs[i]))

    env = env.BulletEnv(env_spec, agents)
    dt, env_info, measurement_groups = env.reset()

    record = []
    print("Simulation progress:")
    for it in progressbar.progressbar(range(100)):
        actions = {}
        for agent in agents:
            actions[agent.name] = agent.action(dt, measurement_groups[agent.name])
        
        dt, env_info, measurement_groups = env.step(actions)
        record.append((env_info,measurement_groups))
        
    evaluator.evaluate(record)