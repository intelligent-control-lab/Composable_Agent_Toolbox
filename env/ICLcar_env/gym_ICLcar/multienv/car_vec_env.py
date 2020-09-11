import numpy as np
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info, obs_to_dict

from gym_ICLcar.envs.wrappers import make_wrapped_env
from gym_ICLcar.multienv.make_env import make_env
from gym_ICLcar.multienv.shmem_vec_env import ShmemVecEnv

def flatten_dict(dicts, keys, copy=False):
  """
  Turns list of dicts into dict of np arrays
  """
  return {
    key: np.array([d[key] for d in dicts], dtype=dicts[0][key].dtype, copy=copy)
    for key in keys
  }

class CarVecEnv(ShmemVecEnv):
  """
  Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
  Specific to car environment and making sure environments are synchronized
  """


  def __init__(self,
    seed,
    env_id,
    num_envs=10,
    **kwargs):

    self.master_env = make_wrapped_env(
      env_id=env_id,
      rank=0,
      **kwargs)

    self.master_env.reset()
    self.master_env.close()
    spaces=(self.master_env.observation_space, self.master_env.action_space)

    env_fns = [make_env(
            env_id=env_id,
            seed=seed,
            rank=indx,
            **kwargs
            ) for indx in range(num_envs)]

    super(CarVecEnv, self).__init__(env_fns=env_fns, spaces=spaces, subproc_worker=car_subproc_worker)


  def step(self, *args, **kwargs):
    obs, reward, done, info = super(CarVecEnv, self).step(*args, **kwargs)
    return obs, reward, done, info

  def reset(self, *args, **kwargs):
    obs = super(CarVecEnv, self).reset(*args, **kwargs)
    return obs

def car_subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
  """
  Control a single environment instance using IPC and
  shared memory.
  """
  def _write_obs(dict_obs):
    flatdict = obs_to_dict(dict_obs)
    for k in dict_obs.keys():
      dst = obs_bufs[k].get_obj()
      dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(
          obs_shapes[k])  # pylint: disable=W0212
      np.copyto(dst_np, flatdict[k])

  env = env_fn_wrapper.x()
  parent_pipe.close()
  try:
    while True:
      cmd, data = pipe.recv()
      if cmd == 'reset':
        pipe.send(_write_obs(env.reset()))
      elif cmd == 'step':
        obs, reward, done, info = env.step(data)
        if done:
          obs = env.reset()

        pipe.send((_write_obs(obs), reward, done, info))

      # ======================================================
      # original
      # ======================================================
      elif cmd == 'render':
        pipe.send(env.render(mode='rgb_array'))
      elif cmd == 'close':
        pipe.send(None)
        break
      # ======================================================
      # added
      # ======================================================
      else:
        raise RuntimeError('Got unrecognized cmd %s' % cmd)
  except KeyboardInterrupt:
    print('ShmemVecEnv worker: got KeyboardInterrupt')
  finally:
    env.close()