import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .random_track_generation import generate_track

from .objects import *

class ICLcarEnv_v2(gym.Env):
    def __init__(self):
        self.canvas = plt.figure()
        self.canvas.canvas.mpl_connect('key_press_event', self._handle_player_events)

    def setup(self, args):
        '''Setup function.
        '''
        self.args = args
        self.define_spaces()

    def define_spaces(self):
        self.action_space = spaces.Dict({
            'action': spaces.Box(low=0, high=self.args['act_limit'], shape=(2,))
        })
        self.observation_space = spaces.Dict({
            # 'pose': spaces.Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(3,)),
            'velocity': spaces.Box(low=-200, high=200, shape=(1,)),
            'acceleration': spaces.Box(low=-200, high=200, shape=(1,)),
            'angular_velocity': spaces.Box(low=-100, high=100, shape=(1,)),
            'angular_acceleration': spaces.Box(low=-200, high=200, shape=(1,)),
            'trans_coef': spaces.Box(low=0, high=100, shape=(1,)),
            'rot_coef': spaces.Box(low=0, high=100, shape=(1,)),
        })

    @property
    def _done(self): return self.done

    @property
    def obs(self):
        obs=dict(
            # pose=self.car.pose,
            velocity=self.car.v,
            acceleration=self.car.dv,
            angular_velocity=self.car.w,
            angular_acceleration=self.car.dw,
            trans_coef=self.car.Bv,
            rot_coef=self.car.Bw
        )
        return obs

    def _handle_player_events(self, event):
        action = [0, 0]

        if event.key == 'a':
            print('here')
            self.car.step([0, 20])
        elif event.key == 'd':
            self.car.step([20, 0])
        elif event.key == 'w':
            self.car.step([100, 100])
        elif event.key == 's':
            self.car.step([-100, -100])
        elif event.key == 'q':
            sys.exit()

        return np.array(action)

    def update_reward(self, step_reward):
        self.reward += step_reward

    def step(self, action, mode='rgb_array'):
        done = False
        step_reward = 0
        info = {}
        import collections
        pose = self.car.pose
        moments = [self.car.v, self.car.w, self.car.dv, self.car.dw]
        friction = [self.car.Bv, self.car.Bw]
        info = collections.OrderedDict(
            action=action,
            pose=pose,
            moments=moments,
            friction=friction,
            reward=self.reward,
            done=done
        )

        self.render()

        return self.obs, step_reward, done, info

    def reset(self):
        self.reward = 0
        self.prev_reward = 0
        self.road, comp, center = generate_track()
        self.car = Car(0, self.road[0][0], self.road[0][1])

        return self.obs

    def render(self, mode='human'):
        plt.cla()
        plt.plot(self.car.center[0], self.car.center[1], marker=(3, 0, self.car.theta_degrees), markersize=10, color='b')
        plt.plot(self.road[:,0], self.road[:,1])
        plt.pause(0.033)
        plt.draw()

