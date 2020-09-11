import os
import torch
import gym
import yaml
import pickle
import copy
import spinup
import torch.nn as nn
import numpy as np
from gym_ICLcar.envs.wrappers import make_wrapped_env

from src.external.actor_critic import ActorCriticDDPG, ActorCriticSAC
from src.external.replay_buffer import DictReplayBuffer
from src.external.tf_logger import TensorboardLogger
from src.utils import *
from spinningup.spinup.utils.run_utils import ExperimentGrid, setup_logger_kwargs
from ray import tune
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def experiment(args):
  if args['hide_display']: os.environ["SDL_VIDEODRIVER"] = "dummy"

  mode = args['mode']
  log(f'Mode: {mode}', color='green')

  # ==================================
  # Logger
  # ==================================
  exp_name = args['exp_name']
  logger_kwargs = setup_logger_kwargs(exp_name, args['seed'], data_dir=args['log_dir'])
  output_dir = logger_kwargs['output_dir']
  dir_exists = os.path.exists(output_dir)

  if dir_exists and mode == 'train':
    log(f'Experiment name {exp_name} already used', 'red')
    raise RuntimeError(f'Experiment name {exp_name} already used')


  if args['experiment_grid']:
    logger_kwargs['output_dir'] = './'

  # ==================================
  # Environment
  # ==================================
  def env_fn(mode):
      if mode == 'test':
        video_kwargs = dict(
          save_frames=args['save_frames'],
          save_dir=os.path.join(output_dir, 'videos'),
          video_save_frequency=args['video_save_frequency']
        )
      else:
        video_kwargs = dict()

      reward_func_kwargs = dict(
        velocity_reward_weight=args['velocity_reward_weight'],
        rotation_penalty_weight=args['rotation_penalty_weight'],
        distance_penalty_weight=args['distance_penalty_weight'],
        angle_penalty_weight=args['angle_penalty_weight'],
        stationary_penalty_weight=args['stationary_penalty_weight']
      )

      env_kwargs = dict(
          env_id=args['environment_id'],
          env_mode=args['env_mode'],
          use_textures=args['use_textures'],
          state_sources=args['state_sources'],
          num_future_info=args['num_future_info'],
          reward_func_kwargs=reward_func_kwargs,
          video_kwargs=video_kwargs
      )
      train_envs = make_wrapped_env(rank=0, **env_kwargs)
      train_envs.setup(args)
      return train_envs

  test_env = env_fn('test')
  # def env_fn(
  #   mode,
  #   velocity_reward_weight,
  #   rotation_penalty_weight,
  #   distance_penalty_weight,
  #   angle_penalty_weight,
  #   stationary_penalty_weight,
  #   env_id,
  #   env_mode,
  #   use_textures,
  #   num_future_info,
  #   save_frames=False,
  #   output_dir='',
  #   video_save_frequency=0,
  # ):
  #   if mode == 'test':
  #     video_kwargs = dict(
  #       save_frames=save_frames,
  #       save_dir=os.path.join(output_dir, 'videos'),
  #       video_save_frequency=video_save_frequency
  #     )
  #   else:
  #     video_kwargs = dict()

  #   reward_func_kwargs = dict(
  #     velocity_reward_weight=velocity_reward_weight,
  #     rotation_penalty_weight=rotation_penalty_weight,
  #     distance_penalty_weight=distance_penalty_weight,
  #     angle_penalty_weight=angle_penalty_weight,
  #     stationary_penalty_weight=stationary_penalty_weight
  #   )

  #   env_kwargs = dict(
  #       env_id=env_id,
  #       env_mode=env_mode,
  #       use_textures=use_textures,
  #       state_sources=state_sources,
  #       num_future_info=num_future_info,
  #       reward_func_kwargs=reward_func_kwargs,
  #       video_kwargs=video_kwargs
  #   )
  #   train_envs = make_wrapped_env(rank=0, **env_kwargs)
  #   train_envs.setup(args)
  #   return train_envs

  # test_env = env_fn(
  #   mode='test',
  #   velocity_reward_weight=args['velocity_reward_weight'],
  #   rotation_penalty_weight=args['rotation_penalty_weight'],
  #   distance_penalty_weight=args['distance_penalty_weight'],
  #   angle_penalty_weight=args['angle_penalty_weight'],
  #   stationary_penalty_weight=args['stationary_penalty_weight'],
  #   environment_id=args['environment_id'],
  #   env_mode=args['env_mode'],
  #   use_textures=args['use_textures'],
  #   num_future_info=args['num_future_info'],
  #   save_frames=args['save_frames'],
  #   output_dir=args['output_dir'],
  #   video_save_frequency=args['video_save_frequency']
  # )

  obs_space, action_space = test_env.observation_space, test_env.action_space
  test_env.close()

  device = torch.device("cuda" if torch.cuda.is_available() and args['use_gpu'] else "cpu")
  log('Running on: {}'.format(device), color='green')

  # ==================================
  # Replay Buffer
  # ==================================
  replay_buffer_kwargs = dict(
    obs_space=obs_space,
    action_space=action_space,
    size=args['buffer_size'],
    device=device
  )

  replay_buffer = DictReplayBuffer

  # ==================================
  # Actor Critic Models
  # ==================================
  encoder_kwargs = dict(
    input_width=args['input_size'],
    input_height=args['input_size'],
    input_channels=args['channels'],
    output_size=args['image_embedding_dimension'],
    kernel_sizes=[4, 4],
    n_channels=[32, 64],
    strides=[2, 2],
    paddings=np.zeros(2, dtype=np.int64),
    hidden_sizes=None,
    added_fc_input_size=0,
    batch_norm_conv=False,
    batch_norm_fc=False,
    init_w=1e-4,
    hidden_init=nn.init.xavier_uniform_,
    hidden_activation=nn.ReLU(),
    # output_activation=identity,
  )

  obs_dim=sum([obs_space.spaces[k].shape[0] for k in obs_space.spaces.keys() if k != 'image'])
  mlp_kwargs = dict(
    sizes=[obs_dim, args['context_embedding_dimension']],
    activation=nn.ReLU()
  )

  ac_kwargs = dict(
    observation_space=obs_space,
    action_space=action_space,
    hidden_sizes=args['hidden_sizes'],
    activation=torch.nn.ReLU,
    encoder_kwargs=encoder_kwargs,
    mlp_kwargs=mlp_kwargs,
    device=device
  )

  # ==================================
  # Algorithm shared arguments
  # ==================================
  shared_kwargs = dict(
    mode=mode,
    ac_kwargs=ac_kwargs,
    replay_buffer=replay_buffer,
    replay_buffer_kwargs=replay_buffer_kwargs,
    seed=args['seed'],
    steps_per_epoch=args['steps_per_epoch'],
    epochs=args['epochs'],
    replay_size=args['buffer_size'],
    gamma=args['gamma'],
    polyak=args['tau'],
    batch_size=args['batch_size'],
    start_steps=args['start_steps'],
    update_after=args['update_after'],
    update_every=args['update_every'],
    num_test_episodes=args['num_test_episodes'],
    max_ep_len=args['max_episode_length'],
    logger=TensorboardLogger,
    logger_kwargs=logger_kwargs,
    save_freq=1,
    device=device
  )

  # ==================================
  # Single experiment
  # ==================================
  if args['algo'] == 'ddpg':
    if args['mode'] == 'train':
      spinup.ddpg_pytorch(
        env_fn,
        actor_critic=ActorCriticDDPG,
        pi_lr=args['actor_lr'],
        q_lr=args['critic_lr'],
        act_noise=args['noise_stddev'],
        **shared_kwargs
      )
  elif args['algo'] == 'sac':
    if args['mode'] == 'train':
      # Save a copy of the argument configs
      os.makedirs(output_dir, exist_ok=True)
      pickle.dump(args, open(os.path.join(output_dir, 'experiment_config.pkl'), 'wb'))

      spinup.sac_pytorch(
        env_fn,
        actor_critic=ActorCriticSAC,
        lr=args['learning_rate'],
        alpha=args['alpha'],
        **shared_kwargs
      )
    else:
      spinup.sac_pytorch_test(
        env_fn,
        checkpoint_file=args['checkpoint_file'],
        actor_critic=ActorCriticSAC,
        **shared_kwargs
      )
  else:
    raise NotImplementedError

def tune_experiment(args):
  exp_grid = yaml.load(open(args['experiment_grid'], 'r'))
  grid_keys = exp_grid['grid'].keys()

  grid_search = copy.deepcopy(args)

  for k, v in exp_grid['grid'].items():
    if isinstance(v, list):
      grid_search[k] = tune.grid_search(v)

  analysis = tune.run(
    experiment,
    config=grid_search,
    resources_per_trial={'cpu': args['num_cpu'], 'gpu': 0.2},
    local_dir=args['log_dir'],
    name=args['exp_name']
  )

if __name__ == "__main__":
  from src.configs import add_experiment_args, add_logging_args, add_training_args, add_rl_agent_args, add_encoder_args, add_car_env_args, add_spinning_up_args
  from pprint import pprint
  import argparse
  import json
  from src.utils import load_experiment_settings
  parser = argparse.ArgumentParser(description='DDPG for CarEnv')
  parser = add_experiment_args(parser)
  parser = add_logging_args(parser)
  parser = add_training_args(parser)
  parser = add_rl_agent_args(parser)
  parser = add_encoder_args(parser)
  parser = add_car_env_args(parser)
  parser = add_spinning_up_args(parser)

  args, unknown = parser.parse_known_args()
  args = vars(args)

  experiment_settings = load_experiment_settings(args['experiment_settings'])
  args.update(experiment_settings)

  pprint(args)

  if args['experiment_grid']:
    tune_experiment(args)
  else:
    if args['mode'] == 'test':
      checkpoint_file = os.path.join(args['base_folder'], 'checkpoints', args['checkpoint_file'])
      exp_params = os.path.join(args['base_folder'], 'params.json')
      if os.path.exists(exp_params):
        log(f'Loading exp params from: {exp_params}')
        exp_args = json.load(open(exp_params, 'r'))
        args.update(exp_args)
        import ipdb; ipdb.set_trace()
      experiment(args)
    else:
      experiment(args)

