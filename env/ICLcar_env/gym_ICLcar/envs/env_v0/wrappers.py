import os
import re
import gym
import cv2
import glob
import math
import torch
import random
from gym import spaces
import numpy as np
import pygame as pg
from .objects import Sensors, Info
from .env_configs import *
from src.utils import *
import ipdb

def make_wrapped_env(
  env_id,
  rank,
  env_mode='human',
  use_textures=False,
  state_sources=['lane_direction', 'center_lane'],
  num_future_info=0,
  reward_func_kwargs=dict(),
  video_kwargs=dict()
):
  env = gym.make(env_id)

  if len(set(state_sources) - set("image")) > 0:
    env = SensorWrapper(env, state_sources, num_future_info)

  if 'image' not in state_sources: # can't do this once we add image to state space
    env = InfoWrapper(env)

  if env_mode == 'rgb_array' and 'image' in state_sources:
    env = ImageWrapper(env)

  env = RewardWrapper(env, reward_func_kwargs)

  if video_kwargs and video_kwargs['save_frames']:
    env = VideoWrapper(env, video_kwargs)

  return env

class EnvWrapperBase(gym.Wrapper):
  def __init__(self, env):
    '''Base wrapper class for environment
    '''
    super().__init__(env)

  def setup(self, args, **kwargs):
    self.env.setup(args, **kwargs)
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space

  def step(self, **kwargs):
    return self.env.step(**kwargs)

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

class SensorWrapper(EnvWrapperBase):
  def __init__(self,
    env,
    state_sources,
    num_future_info
  ):
    super().__init__(env)

    self.state_sources = state_sources
    self.num_future_info = num_future_info

  def setup(self, args, **kwargs):
    super(SensorWrapper, self).setup(args, **kwargs)
    self.define_spaces()

  def define_spaces(self):
    if 'center_lane' in self.state_sources:
      self.env.observation_space.spaces['dist_to_center_lane'] = spaces.Box(low=-1000, high=1000, shape=(1,))
      self.env.observation_space.spaces['closest_center_lane_pose'] = spaces.Box(low=-1000, high=1000, shape=(2,))
      if self.num_future_info > 0:
        self.env.observation_space.spaces['future_center_pos_car_coord'] = spaces.Box(low=-1000, high=1000, shape=(2 * self.num_future_info,))

    if 'lane_curvature' in self.state_sources:
      self.env.observation_space.spaces['lane_curvature'] = spaces.Box(low=0, high=100, shape=(1,))

    range_sensor = list(filter(lambda x: bool(re.match('range_.*', x)), self.state_sources))
    if range_sensor:
      self.num_range_sensors = int(range_sensor[0].split('_')[-1])
      for i in range(self.num_range_sensors):
        self.env.observation_space.spaces[f'range_{i}'] = spaces.Box(low=0, high=100, shape=(1,))

    if 'lane_direction' in self.state_sources:
      self.env.observation_space.spaces['angle_diff'] = spaces.Box(low=-math.pi, high=math.pi, shape=(1,))
      if self.num_future_info > 0:
        self.env.observation_space.spaces['future_lane_direction'] = spaces.Box(low=-math.pi, high=math.pi, shape=(self.num_future_info,))

  def add_sensor_to_obs(self, obs):
    for i, sensor in enumerate(self.sensors.range_sensors):
        obs[f'range_{i}'] = sensor.measurement

    if 'center_lane' in self.state_sources:
      obs['dist_to_center_lane'] = self.sensors.center_lane_sensor.measurement[0]
      obs['closest_center_lane_pose'] = self.sensors.center_lane_sensor.measurement[1:3]
      if self.num_future_info > 0:
        obs['future_center_pos_car_coord'] = self.sensors.center_lane_sensor.measurement[1:]

    if 'lane_curvature' in self.state_sources:
      obs['lane_curvature'] = self.sensors.lane_curvature_sensor.measurement

    if 'lane_direction' in self.state_sources:
      # obs['lane_direction'] = self.sensors.lane_direction_sensor.measurement[0]
      obs['angle_diff'] = self.sensors.lane_direction_sensor.measurement[0]
      if self.num_future_info > 0:
        obs['future_lane_direction'] = self.sensors.lane_direction_sensor.measurement[1:]
    return obs

  def reset(self):
    obs = self.env.reset()
    self.sensors = Sensors(self.env.car, self.env.road, self.state_sources, self.num_future_info)

    self.add_sensor_to_obs(obs)
    return obs

  def step(self, action, mode='rgb_array'):
    obs, reward, done, info = self.env.step(action, mode)
    self.sensors.blit(self.env.screen)
    self.add_sensor_to_obs(obs)

    for sensor in self.sensors.sensors:
      if sensor.name == 'LaneDirectionSensor':
        info.update({
          # 'road_direction': sensor.measurement[0], # info only visualizes the current direction
          'angle_diff': sensor.measurement[0],
        })
      elif sensor.name == 'CenterLaneSensor0':
        info.update({'closest_dist': sensor.measurement[0]})

    return obs, reward, done, info


class InfoWrapper(EnvWrapperBase):
  def __init__(self, env):
    super().__init__(env)

  def setup(self, args, **kwargs):
    super(InfoWrapper, self).setup(args, **kwargs)

  def reset(self):
    obs = self.env.reset()
    font_size = int(self.env.screen.get_size()[1] / 30)
    self.info = Info(font_size=font_size)
    self.info.set_sensors(self.env.sensors)

    return obs

  def step(self, action, mode='rgb_array'):
    obs, reward, done, info = self.env.step(action, mode)
    self.info.blit(self.env.screen, info)
    return obs, reward, done, info


class RewardWrapper(EnvWrapperBase):
  def __init__(self, env, reward_func_kwargs):
    super().__init__(env)
    self.reward_func_kwargs=reward_func_kwargs

  def reset(self):
    obs = self.env.reset()
    return obs

  def step(self, action, mode='rgb_array'):
    obs, _, done, info = self.env.step(action, mode)
    step_reward, done = self.reward_func(action, info)

    info['reward'] = self.reward
    return obs, step_reward, done, info

  def reward_func(self, action, info):
    done = False
    step_reward = 0

    # Car goes out of map
    # if self.out_of_bound_check():
    #   done = True
    #   step_reward = -10
    # else:
    angle_diff = info['angle_diff']
    dist_to_center = info['closest_dist']

    velocity = self.car.v
    angular_vel = self.car.w

    # enforces that the car should go in the correct direction
    min_velocity_rew = -40
    velocity_rew = self.reward_func_kwargs['velocity_reward_weight'] * self.car.v * math.cos(angle_diff)
    velocity_rew = max(min_velocity_rew, velocity_rew)
    step_reward += velocity_rew

    # Penalty for being far way from center of lane
    max_dist_to_center_penalty = 4
    dist_to_center_threshold = 30
    if abs(dist_to_center) > dist_to_center_threshold:
      dist_to_center_penalty = (self.reward_func_kwargs['distance_penalty_weight']*(abs(dist_to_center) - dist_to_center_threshold))**2
      dist_to_center_penalty = min(dist_to_center_penalty, max_dist_to_center_penalty)
      step_reward -= dist_to_center_penalty

    # Penalty for high angular velocity
    max_angular_vel_penalty = 2
    angular_vel_threshold = 3
    if abs(angular_vel) > angular_vel_threshold:
      angular_vel_penalty = (self.reward_func_kwargs['rotation_penalty_weight']*(abs(angular_vel) - angular_vel_threshold))**2
      angular_vel_penalty = min(angular_vel_penalty, max_angular_vel_penalty)
      step_reward -= angular_vel_penalty

    stationary_penalty = 0.01
    step_reward -= self.reward_func_kwargs['stationary_penalty_weight']*stationary_penalty

    # print(f"vlon: {velocity_rew/30}, dist: {dist_to_center_penalty}, ang_vel: {angular_vel_penalty}, ang_diff: {diff_angle_penalty}")

    self.update_reward(step_reward)
    return step_reward, done


class ImageWrapper(EnvWrapperBase):
  def __init__(self, env):
    super().__init__(env)

  def setup(self, args, **kwargs):
    super(ImageWrapper, self).setup(args, **kwargs)
    self.define_spaces()

  def define_spaces(self):
    self.env.observation_space.spaces['image'] = gym.spaces.Box(
      low=0, high=255, shape=(self.args['input_size'], self.args['input_size'], self.args['channels']), dtype=np.uint8
      )

  def add_image_to_obs(self, obs):
    obs['image'] = pg.surfarray.pixels3d(self.env.screen)

    # RGB to grayscale
    if obs['image'].shape[2] != self.args['channels']:
      obs['image'] = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2GRAY)
      obs['image'] = obs['image'][:, :, np.newaxis]

    # resize image to input size
    if self.args['input_size'] != obs['image'].shape[:2]:
      obs['image'] = cv2.resize(obs['image'], (self.args['input_size'], self.args['input_size']), interpolation=cv2.INTER_LINEAR)

    # add channel dimension
    if len(obs['image'].shape) == 2:
      obs['image'] = np.expand_dims(obs['image'], 2)

  def reset(self):
    obs = self.env.reset()
    self.add_image_to_obs(obs)
    return obs

  def step(self, action, mode='rgb_array'):
    obs, reward, done, info = self.env.step(action, mode)
    self.add_image_to_obs(obs)
    return obs, reward, done, info


class VideoWrapper(EnvWrapperBase):
  '''
    Creates a video of the trajectory generated by the current policy
    for visualization.
    Output: log_dir/video01/frames0000.png
  '''
  def __init__(self, env, video_kwargs=dict()):
    super().__init__(env)
    self.frame_ctr = 0
    self._start_saving = False
    self.video_ctr = -1
    self.episode = 0
    self.save_dir = video_kwargs['save_dir']
    self.save_freq = video_kwargs['video_save_frequency']

  def set_save_dir(self, save_dir):
    self.save_dir = save_dir

  def step(self, action, mode='rgb_array'):
    obs, reward, done, info = self.env.step(action, mode)
    if self._start_saving:
      img = self.env.screen
      img = pg.transform.scale(img, (600, 600))
      _ctr = str(self.frame_ctr).zfill(4)
      fn = os.path.join(self.save_dir, f"video_{self.video_ctr}/", f"frame_{_ctr}.png")
      pg.image.save(img, fn)
      self.frame_ctr += 1

    return obs, reward, done, info

  def reset(self):
    obs = self.env.reset()

    if self.episode % self.save_freq == 0:
      self._start_saving = True
      self.frame_ctr = 0
      self.video_ctr += 1
      video_dir = os.path.join(self.save_dir, f"video_{self.video_ctr}")
      print(f'Created {video_dir}')
      os.makedirs(video_dir)
      self.video_dir = video_dir
    else:
      self.create_video()
      self._start_saving = False

    self.episode += 1

    return obs

  def create_video(self):
    if self._start_saving:
      print(f"Created {self.video_dir}/trajectory.mp4")
      os.system(f"ffmpeg -loglevel quiet -r 30 -i {self.video_dir}/frame_%4d.png -pix_fmt yuv420p {self.video_dir}/trajectory.mp4")

      # Remove the png files
      png_files = glob.glob(f'{self.video_dir}/*.png')
      for f in png_files:
        os.remove(f)