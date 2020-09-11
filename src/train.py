import gym
import random
import numpy as np
import argparse
import gym_ICLcar
from gym_ICLcar.envs.wrappers import make_wrapped_env
from copy import deepcopy
from collections import namedtuple
import ipdb

import torch
from src.ddpg import DDPG
from src.utils import *
from src.modules.replay_buffer import ReplayBuffer, Transition
from src.modules.noise import OrnsteinUhlenbeckActionNoise


def train(
    agent,
    replay_buffer,
    env,
    num_iterations,
    max_episode_length,
    control_frequency,
    verbose=False,
    render=False
    ):

    total_steps = 0
    episode_num = 0
    episode_steps = 0
    episode_reward = 0

    state = env.reset()

    while total_steps < num_iterations:
        action = agent.select_action(state)

        dt = env.clock.tick_busy_loop(control_frequency) / 1000.0 # in milliseconds
        next_state, reward, done, info = env.step(action.detach().numpy(), dt)

        if render:
            env.render()

        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        replay_buffer.append(torch.Tensor(state), torch.Tensor(action), torch.Tensor(next_state), torch.Tensor([reward]), torch.Tensor([int(done)]))

        transitions = replay_buffer.sample()

        if transitions:
            batch = Transition(*zip(*transitions))
            actor_loss, critic_loss = agent.update_policy(batch)

        # =====================
        # Increment counters
        # =====================
        total_steps += 1
        episode_steps += 1
        episode_reward += reward
        state = next_state

        # =====================
        # End of episode
        # =====================
        if done:
            if verbose:
                if (episode_num % args['log_frequency']) == 0:
                    print('Episode: {} | Episode reward: {} | Episode steps: {} | Actor loss: {} | Critic loss: {}'.format(episode_num, episode_reward, episode_steps, actor_loss, critic_loss))

            state = env.reset()
            episode_num += 1
            episode_steps = 0
            episode_reward = 0

def experiment(args):
    # ==================================
    # Environment
    # ==================================
    env_kwargs = dict(
        env_id=args['environment_id'],
        seed=args['seed'],
        env_mode=args['env_mode'],
        num_range_sensors=args['num_range_sensors'],
        center_lane_sensor=args['center_lane_sensor'],
        lane_curvature_sensor=args['lane_curvature_sensor']
    )
    if args['num_envs'] > 1:
        from gym_ICLcar.multienv.car_vec_env import CarVecEnv
        train_envs = CarVecEnv(**env_kwargs)
    else:
        del env_kwargs['seed']
        train_envs = make_wrapped_env(rank=0, **env_kwargs)
    import ipdb; ipdb.set_trace()
    # ==================================
    # Replay buffer
    # ==================================
    buffer_kwargs=dict(
        batch_size=args['batch_size'],
        buffer_size=args['buffer_size']
    )
    replay_buffer = ReplayBuffer(**buffer_kwargs)

    # ==================================
    # Agent
    # ==================================
    num_actions = train_envs.action_space['action'].shape[0]
    exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(num_actions), sigma=args['noise_stddev'] * np.ones(num_actions))

    ddpg_kwargs=dict(
        tau=args['tau'],
        actor_lr=args['actor_lr'],
        critic_lr=args['critic_lr'],
        discount=args['discount'],
        uniform_init=args['uniform_init'],
        hidden_sizes=args['hidden_sizes'],
        observation_space=train_envs.observation_space,
        action_space=train_envs.action_space,
        exploration_noise=exploration_noise
    )
    agent = DDPG(**ddpg_kwargs)

    # ==================================
    # Training
    # ==================================
    training_kwargs=dict(
        agent=agent,
        replay_buffer=replay_buffer,
        env=train_envs,
        num_iterations=args['num_iterations'],
        max_episode_length=args['max_episode_length'],
        control_frequency=args['control_frequency'],
        verbose=args['verbose'],
        render=args['render']
    )

    if args['mode'] == 'train':
        train(**training_kwargs)
    elif args['mode'] == 'test':
        raise NotImplementedError
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


if __name__ == "__main__":
    from src.configs import add_experiment_args, add_training_args, add_rl_agent_args, add_car_env_args, add_spinning_up_args
    from pprint import pprint
    import argparse
    from src.utils import load_experiment_settings
    parser = argparse.ArgumentParser(description='Custom DDPG for CarEnv')
    parser = add_experiment_args(parser)
    parser = add_training_args(parser)
    parser = add_rl_agent_args(parser)
    parser = add_car_env_args(parser)
    parser = add_spinning_up_args(parser)
    args, unknown = parser.parse_known_args()
    args = vars(args)

    experiment_settings = load_experiment_settings(args['experiment_settings'])
    args.update(experiment_settings)

    pprint(args)
    main(args)
    experiment(args)