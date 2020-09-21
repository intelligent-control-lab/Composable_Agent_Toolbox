import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .policy_base import PolicyBase
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the Actor network
class Actor(nn.Module):
    def __init__(self, env_params):
        super(Actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class Critic(nn.Module):
    def __init__(self, env_params):
        super(Critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class ActorCriticPolicy(PolicyBase):

    def __init__(self, policy_spec):

        model_path = policy_spec["model_path"]
        o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor = Actor(policy_spec["env_params"])
        self.actor.load_state_dict(model)
        self.actor.eval()

    def action(self, dt, sensors_data):
        inputs = np.vstack([sensors_data["state_sensor"]["state"], sensors_data["cartesian_sensor"]["pos"], sensors_data["goal_sensor"]["rel_pos"]])
        inputs = torch.tensor(inputs.T, dtype=torch.float32)

        with torch.no_grad():
            pi = self.actor(inputs)
        
        control = pi.detach().numpy().squeeze()
        control = control[:3]

        return control