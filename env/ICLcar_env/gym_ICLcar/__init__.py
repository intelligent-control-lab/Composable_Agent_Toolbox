from gym.envs.registration import register
import gym

found = False
for env in gym.envs.registry.env_specs:
  if 'ICLcar-v0' in env:
    found = True

if not found:
  register(
      id='ICLcar-v0',
      entry_point='gym_ICLcar.envs:ICLcarEnv',
  )

found = False
for env in gym.envs.registry.env_specs:
  if 'ICLcar-v1' in env:
    found = True

if not found:
  register(
      id='ICLcar-v1',
      entry_point='gym_ICLcar.envs:ICLcarEnv_v2',
  )

found = False
for env in gym.envs.registry.env_specs:
  if 'ICLcar-v2' in env:
    found = True

if not found:
  register(
      id='ICLcar-v2',
      entry_point='gym_ICLcar.envs:ICLcarEnv_v3',
  )