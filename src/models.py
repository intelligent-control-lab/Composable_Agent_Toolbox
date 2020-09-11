import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(tensor, fan_in=None):
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

class Actor(nn.Module):
    """Defines actor network"""
    def __init__(
        self,
        num_inputs,
        num_actions,
        uniform_init=[3e-3, 3e-4],
        hidden_sizes=[400, 300]
    ):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.layer_norm1 = nn.LayerNorm(hidden_sizes[0])

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer_norm2 = nn.LayerNorm(hidden_sizes[1])

        self.out = nn.Linear(hidden_sizes[1], num_actions)

        # ==================
        # Layer weights init
        # ==================
        fanin_init(self.fc1.weight)
        fanin_init(self.fc1.bias)

        fanin_init(self.fc2.weight)
        fanin_init(self.fc2.bias)

        nn.init.uniform_(self.out.weight, -uniform_init[0], uniform_init[0])
        nn.init.uniform_(self.out.bias, -uniform_init[1], uniform_init[1])

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        out = torch.tanh(self.out(x))
        return out

class Critic(nn.Module):
    """Defines critic network"""
    def __init__(
        self,
        num_inputs,
        num_actions,
        uniform_init=[3e-3, 3e-4],
        hidden_sizes=[400, 300]
    ):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.layer_norm1 = nn.LayerNorm(hidden_sizes[0])

        self.fc2 = nn.Linear(hidden_sizes[0] + num_actions, hidden_sizes[1])
        self.layer_norm2 = nn.LayerNorm(hidden_sizes[1])

        self.value = nn.Linear(hidden_sizes[1], 1)

        # ==================
        # Layer weights init
        # ==================
        fanin_init(self.fc1.weight)
        fanin_init(self.fc1.bias)

        fanin_init(self.fc2.weight)
        fanin_init(self.fc2.bias)

        nn.init.uniform_(self.value.weight, -uniform_init[0], uniform_init[0])
        nn.init.uniform_(self.value.bias, -uniform_init[1], uniform_init[1])

    def forward(self, inputs, actions):
        x = self.fc1(inputs)
        x = self.layer_norm1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        value = self.value(x)
        return value

