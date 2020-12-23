"""This is the physics engine module. Will be replaced by other physics engine in the future.

The World class is a simple physics engine. It will instantiate agents and sensors
define by users, and simulate the interactions. 

The World class is called by the Envrionment class.
"""

import numpy as np
from env.base_world.world import World
import importlib
import env.flat_world.agent

class FlatReachingWorld(World):
    """Reaching goal scenario.

    Each agent in this world has a goal. Once the agent reaches the goal,
    the goal will move to a new place.

    The goal is represented by a virtual agent. Such agent has no collision volume.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.goal_agents = []
        self.reaching_eps = spec["reaching_eps"]
        self.agent_goal_lists = spec["agent_goal_lists"]
        
    def add_agent(self, comp_agent, agent_env_spec):
        """Instantiate an agent and its goal in the env.

        This function instantiate a user defined agents in the physical world. 
        And sensors attached to this agent will also be instantiated by calling  _add_sensor.
        A static goal agent will also be instantiated into the world. The goal agent only
        moves when the agent reaches the goal.

        Args:
            comp_agent: the computational agent object which decides the behavior of the agent.
                This object also constains the specs of the agent, such as name, sensor types, etc.
            agent_env_spec: initial position of the physical agent in the simulation environment, etc.
        """
        
        agent = super().add_agent(comp_agent, agent_env_spec)
        
        goal_agent = env.flat_world.agent.GoalAgent(agent.name+"_goal", agent, self.agent_goal_lists[agent.name], self.reaching_eps)
        agent.goal = goal_agent
        self.agents[goal_agent.name] = goal_agent

        return agent, goal_agent  # just in case subclasses call super() and need these.
  
    def simulate(self, actions, dt):
        """One step simulation in the physics engine

        Args:
            actions: the actions taken by the computational agents.
            dt: time separation between two steps.
        
        Returns:
            environment infomation: contains information needed by rendering and evaluator.
            measurement_groups: data of all sensors grouped by agent names.
        """
        for name, agent in self.agents.items():
            if name in actions:
                agent.forward(actions[name], dt)
            else:
                agent.forward()
            
        