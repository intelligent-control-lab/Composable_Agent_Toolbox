import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.models import Actor, Critic
from src.modules.replay_buffer import ReplayBuffer
from src.modules.noise import OrnsteinUhlenbeckActionNoise
from src.utils import *

class DDPG(object):
    def __init__(
        self,
        tau,
        actor_lr,
        critic_lr,
        discount,
        hidden_sizes,
        uniform_init,
        observation_space,
        action_space,
        exploration_noise,
        device='cpu',
    ):

        self.tau = tau
        self.discount = discount
        self.device = device
        self.exploration_noise = exploration_noise
        self.observation_space = observation_space
        self.action_space = action_space
        self.input_size = sum([observation_space.spaces[k].shape[0] for k in observation_space.spaces.keys()])
        self.num_actions = action_space['action'].shape[0]

        self.actor = Actor(self.input_size, self.num_actions, hidden_sizes=hidden_sizes, uniform_init=uniform_init).to(device)
        self.actor_target = Actor(self.input_size, self.num_actions, hidden_sizes=hidden_sizes, uniform_init=uniform_init).to(device)
        self.actor_optimizer  = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(self.input_size, self.num_actions, hidden_sizes=hidden_sizes, uniform_init=uniform_init).to(device)
        self.critic_target = Critic(self.input_size, self.num_actions, hidden_sizes=hidden_sizes, uniform_init=uniform_init).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state):
        input_ = torch.Tensor(state)
        logits = self.actor(input_)

        # rescale logits back after tanh
        logits = logits * self.action_space['action'].high[0]

        noise = self.exploration_noise()
        noise = torch.Tensor(noise)
        logits += noise

        # Clip the output according to the action space of the env
        action = logits.clamp(self.action_space['action'].low[0], self.action_space['action'].high[0])
        return action

    def update_policy(self, batch):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        state_batch = Variable(torch.stack(list(state_batch))).to(self.device)
        action_batch = Variable(torch.stack(list(action_batch))).to(self.device)
        next_state_batch = Variable(torch.stack(list(next_state_batch))).to(self.device)
        reward_batch = Variable(torch.stack(list(reward_batch))).to(self.device)
        done_batch = Variable(torch.stack(list(done_batch))).to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # =====================
        # Compute target Q-values
        # =====================
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.discount * next_state_action_values

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # =====================
        # Update critic
        # =====================
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # =====================
        # Update actor
        # =====================
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # =====================
        # Update target networks
        # =====================
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss.item(), value_loss.item()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()