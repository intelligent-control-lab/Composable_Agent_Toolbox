import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlkit.torch.conv_networks import CNN

from src.utils import *


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, device):
        super().__init__()
        self.device = device
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Sigmoid).to(device)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        context_keys = obs.keys()
        context = [torch.as_tensor(obs[k], dtype=torch.float32, device=self.device) for k in context_keys if k != 'image']

        if len(context[0].shape) > 1:
            context = torch.cat(context, dim=1)
        else:
            context = [c.unsqueeze(0) if len(c.shape) == 0 else c for c in context]
            context = torch.cat(context, dim=0).unsqueeze(0).to(self.device)

        # tanh outputs between [-1, 1], convert to [0, 1] range
        out = self.pi(context).squeeze(0)
        # out = (out + 1) / 2
        return self.act_limit * out

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        self.device = device
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation).to(device)

    def forward(self, obs, act):
        context_keys = obs.keys()
        context = [torch.as_tensor(obs[k], dtype=torch.float32, device=self.device) for k in context_keys if k != 'image']

        if len(context[0].shape) > 1:
            context = torch.cat(context, dim=1)
        else:
            context = [c.unsqueeze(0) if len(c.shape) == 0 else c for c in context]
            context = torch.cat(context, dim=0).unsqueeze(0).to(self.device)
        q = self.q(torch.cat([context, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class CNNMLPActor(nn.Module):

    def __init__(self, image_encoder, context_encoder, hidden_sizes, obs_dim, act_dim, activation, act_limit, device):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit
        self.image_encoder = image_encoder
        self.context_encoder = context_encoder
        self.device = device

    def forward(self, obs):
        # Encode image
        img = obs['image']
        if len(img.shape) == 4:
            img_flatten = img.view(img.shape[0], -1)
        else:
            img = torch.as_tensor(img, dtype=torch.float32, device=self.device)
            img_flatten = img.flatten().unsqueeze(0)

        # B x H x W x C
        image_embedding = self.image_encoder(img_flatten)

        # Encode context (acceleration, velocity, pose, etc)
        context_keys = obs.keys()
        context = [torch.as_tensor(obs[k], dtype=torch.float32, device=self.device)  for k in context_keys if k != 'image']

        if len(context[0].shape) > 1:
            context = torch.cat(context, dim=1)
        else:
            context = [c.unsqueeze(0) if len(c.shape) == 0 else c for c in context]
            context = torch.cat(context, dim=0).unsqueeze(0).to(self.device)

        context_embedding = self.context_encoder(context)

        image_and_context_embedding = torch.cat((image_embedding, context_embedding), dim=1)

        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(image_and_context_embedding).squeeze(0)

class CNNMLPQFunction(nn.Module):

    def __init__(self, image_encoder, context_encoder, hidden_sizes, obs_dim, act_dim, activation, device):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.image_encoder = image_encoder
        self.context_encoder = context_encoder
        self.device = device

    def forward(self, obs, act):
        if not 'image' in obs:
            raise RuntimeError

        # Encode image
        img = obs['image']
        if len(img.shape) == 4:
            img_flatten = img.view(img.shape[0], -1)
        else:
            img = torch.as_tensor(img, dtype=torch.float32, device=self.device)
            img_flatten = img.flatten().unsqueeze(0)

        # B x H x W x C
        image_embedding = self.image_encoder(img_flatten)

        # Encode context (acceleration, velocity, pose, etc)
        context_keys = obs.keys()
        context = [torch.as_tensor(obs[k], dtype=torch.float32, device=self.device) for k in context_keys if k != 'image']

        if len(context[0].shape) > 1:
            context = torch.cat(context, dim=1)
        else:
            context = [c.unsqueeze(0) if len(c.shape) == 0 else c for c in context]
            context = torch.cat(context, dim=0).unsqueeze(0).to(self.device)

        context_embedding = self.context_encoder(context)

        # B x (image_dim + context_dim)
        image_and_context_embedding = torch.cat((image_embedding, context_embedding), dim=1)

        q = self.q(torch.cat([image_and_context_embedding, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ImageEncoder(nn.Module):

    def __init__(self, encoder_kwargs={}):
        super().__init__()
        self.model = CNN(**encoder_kwargs)

    def forward(self, obs, **kwargs):
        embedding = self.model(obs)
        return embedding

class ActorCriticDDPG(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes, activation, device='cpu', encoder_kwargs={}, mlp_kwargs={}):
        super().__init__()
        act_dim = action_space['action'].shape[0]
        act_limit = action_space['action'].high[0]

        # build image encoder
        if 'image' in observation_space:
            obs_dim = encoder_kwargs['output_size'] + mlp_kwargs['sizes'][-1]
            self.image_encoder = ImageEncoder(encoder_kwargs)
            self.context_encoder = mlp(**mlp_kwargs)
        else:
            obs_dim=sum([observation_space.spaces[k].shape[0] for k in observation_space.spaces.keys() if k != 'image'])

        # build policy and value functions
        if 'image' in observation_space:
            self.pi = CNNMLPActor(self.image_encoder, self.context_encoder, hidden_sizes, obs_dim, act_dim, activation, act_limit, device)
            self.q = CNNMLPQFunction(self.image_encoder, self.context_encoder, hidden_sizes, obs_dim, act_dim, activation, device)
        else:
            self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, device)
            self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, device)

        log('Policy network: ', 'green')
        log(self.pi, 'yellow')
        log('Q network: ', 'green')
        log(self.q, 'yellow')

    @property
    def __name__(self):
        return 'ActorCriticDDPG'

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

# ============================
# SAC stuff
# ============================
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, device):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.device = device

    def forward(self, obs, deterministic=False, with_logprob=True):
        # Return output from network scaled to action space limits.
        context_keys = obs.keys()
        context = [torch.as_tensor(obs[k], dtype=torch.float32, device=self.device) for k in context_keys if k != 'image']

        if len(context[0].shape) > 1:
            context = torch.cat(context, dim=1)
        else:
            context = [c.unsqueeze(0) if len(c.shape) == 0 else c for c in context]
            context = torch.cat(context, dim=0).unsqueeze(0).to(self.device)

        net_out = self.net(context)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.sigmoid(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action.squeeze(0), logp_pi

class ActorCriticSAC(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes, activation, device='cpu', encoder_kwargs={}, mlp_kwargs={}):
        super().__init__()

        act_dim = action_space['action'].shape[0]
        act_limit = action_space['action'].high[0]

        # build image encoder
        if 'image' in observation_space:
            obs_dim = encoder_kwargs['output_size'] + mlp_kwargs['sizes'][-1]
            self.image_encoder = ImageEncoder(encoder_kwargs)
            self.context_encoder = mlp(**mlp_kwargs)
        else:
            obs_dim=sum([observation_space.spaces[k].shape[0] for k in observation_space.spaces.keys() if k != 'image'])

        # build policy and value functions
        if 'image' in observation_space:
            # TODO: fix this
            self.pi = CNNMLPActor(self.image_encoder, self.context_encoder, hidden_sizes, obs_dim, act_dim, activation, act_limit, device)
            self.q = CNNMLPQFunction(self.image_encoder, self.context_encoder, hidden_sizes, obs_dim, act_dim, activation, device)
        else:
            # build policy and value functions
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, device)
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, device)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, device)

        log('Policy network: ', 'green')
        log(self.pi, 'yellow')
        log('Q1 network: ', 'green')
        log(self.q1, 'yellow')
        log('Q2 network: ', 'green')
        log(self.q2, 'yellow')

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()