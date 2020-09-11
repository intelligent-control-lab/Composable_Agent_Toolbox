import numpy as np
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class DictReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents that have gym.Dict for
    observation and action space
    """

    def __init__(self, obs_space, action_space, size, device):
        self.obs_keys = list(obs_space.spaces.keys())
        self.act_keys = list(action_space.spaces.keys())

        self.obs_buf = {k: np.zeros(combined_shape(size, obs_space[k].shape), dtype=np.float32) for k in self.obs_keys}
        self.obs2_buf = {k: np.zeros(combined_shape(size, obs_space[k].shape), dtype=np.float32) for k in self.obs_keys}
        self.act_buf = np.zeros(combined_shape(size, action_space['action'].shape), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    @property
    def __name__(self):
        return 'DictReplayBuffer'

    def store(self, obs, act, rew, next_obs, done):
        for k in self.obs_keys:
            self.obs_buf[k][self.ptr] = obs[k]
            self.obs2_buf[k][self.ptr] = next_obs[k]

        self.act_buf[self.ptr] = act

        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs={k: self.obs_buf[k][idxs] for k in self.obs_keys},
                     obs2={k: self.obs2_buf[k][idxs] for k in self.obs_keys},
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        out = {}
        from collections import defaultdict
        for k, v in batch.items():
            if isinstance(v, dict):
                out[k] = defaultdict(dict)
                for k_, v_ in v.items():
                    out[k][k_] = torch.as_tensor(v_, dtype=torch.float32).to(self.device)
            else:
                out[k] = torch.as_tensor(v, dtype=torch.float32).to(self.device)
        return out