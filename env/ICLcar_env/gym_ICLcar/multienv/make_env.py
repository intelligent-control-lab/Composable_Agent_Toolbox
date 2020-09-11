import random
import numpy as np
from gym_ICLcar.envs.wrappers import make_wrapped_env
import torch

def make_env(
  env_id,
  seed,
  rank,
  **kwargs,
  ):

  def _thunk():
      # config['seed'] = seed+rank
      np.random.seed(seed+rank)
      torch.manual_seed(seed+rank)
      random.seed(seed+rank)

      env = make_wrapped_env(env_id, rank, **kwargs)
      return env

  return _thunk