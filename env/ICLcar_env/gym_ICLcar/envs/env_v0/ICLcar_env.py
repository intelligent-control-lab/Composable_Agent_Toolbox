import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import pygame as pg
import numpy as np
from ple.games import base

from .objects import *
from .env_configs import *


class ICLcarEnv(gym.Env, base.PyGameWrapper):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        pg.init()
        self.legacy_screen = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        pg.display.set_caption(CAPTION)

    def setup(self, args):
        '''
            Setup function.
        '''
        if args['env_mode'] == 'human' or args['env_mode'] == 'data_collection':
            self.width, self.height = SCREEN_WIDTH, SCREEN_HEIGHT
        else:
            self.width, self.height = args['crop_size'], args['crop_size']

        base.PyGameWrapper.__init__(self, self.width, self.height)
        self.screen = pg.display.set_mode(self.getScreenDims(), 0, 32)
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

    def _handle_player_events(self):
        action = [0, 0]

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            action = [0, LATERAL_FORCE]
        if keys[pg.K_RIGHT]:
            action = [LATERAL_FORCE, 0]
        if keys[pg.K_UP]:
            action = [FORWARD_FORCE, FORWARD_FORCE]
        if keys[pg.K_DOWN]:
            action = [-FORWARD_FORCE, -FORWARD_FORCE]

        return np.array(action)

    def out_of_bound_check(self):
        return not (-300 < self.car.x < SCREEN_WIDTH + 300) or not (-300 < self.car.y < SCREEN_HEIGHT + 300)

    def collision_check(self):
        pose = self.car.pose[:2]
        pose = [int(ele) for ele in pose]
        return self.road.field_mask.overlap(self.car.img_mask, pose)

    def update_friction(self):
        pose = [int(self.car.x), int(self.car.ypos)]

        if not self.car.img_mask:
            print('dont update')
            return

        set_friction = False

        # Set rotational and translational coefficients of vehicle
        for texture, dic in self.road.texture_map.items():
            if self.road.texture_map[texture]['mask'].overlap(self.car.img_mask, pose):
                self.car.set_friction(dic['friction_level'], 1)
                set_friction = True
                break

        if not set_friction: self.car.set_friction(1, 1)

    def world2screen(self, world_x, world_y):
        screen_x = (world_x - self.world_offset_x) * self.zoom
        screen_y = (world_y - self.world_offset_y) * self.zoom
        return [screen_x, screen_y]


    def screen2world(self, screen_x, screen_y):
        world_x = (screen_x / self.zoom) + self.world_offset_x
        world_y = (screen_y / self.zoom) + self.world_offset_y
        return [world_x, world_y]

    def rotate_surf(self, surf, angle, pivot, offset):
        rotated_image = pg.transform.rotozoom(surf, angle, 1)
        rotated_rect = rotated_image.get_rect(center=pivot+offset)
        return rotated_image, rotated_rect

    def rotate_point(self, point, origin, degrees):
        radians = np.deg2rad(degrees)
        x, y = point
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy

    def blit(self, mode='rgb_array'):
        if mode == 'rgb_array':
            color = self.legacy_screen.get_at((0, 0))
            w, h = self.legacy_screen.get_width(), self.legacy_screen.get_height()

            origin = (w//2, h//2)
            offset = pg.math.Vector2(w//2, h//2)
            angle = 90 - self.car.theta_degrees

            # rotate the screen image to align with car heading
            rotated_surf, rotated_rect = self.rotate_surf(self.legacy_screen, angle, origin, offset)

            # get rotated coordinate or center of car
            rot_x, rot_y = self.rotate_point((self.car.xpos, self.car.ypos), origin, angle)

            # blit on larger screen
            tmp_screen = pg.Surface((SCREEN_WIDTH*2, SCREEN_HEIGHT*2))
            tmp_screen.blit(rotated_surf, (rotated_rect))
            pixelarr = pg.PixelArray(tmp_screen)
            pixelarr.replace(BLACK, color)
            tmp_screen = pixelarr.make_surface()

            self.screen.blit(tmp_screen, (0, 0), (rot_x+offset[0]+CAR_WIDTH-(1/2)*self.width, rot_y+offset[1]-CAR_LENGTH-(2/3)*(self.height), self.width, self.height))
        elif mode == 'human' or mode == 'data_collection':
            self.screen.blit(self.legacy_screen, (0, 0))

    def update_reward(self, step_reward):
        self.reward += step_reward

    def step(self, action, mode='rgb_array'):
        self.screen.fill(WHITE)

        if mode == 'human' or action is None:
            action = self._handle_player_events()

        # ==============================================
        # Update friction and update all of the env objects
        # ==============================================
        self.update_friction()
        self.road.blit(self.legacy_screen)
        self.car.blit(self.legacy_screen, action)

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

        self.blit(mode)

        return self.obs, step_reward, done, info

    def reset(self):
        self.reward = 0
        self.prev_reward = 0
        track_num = self.args['track_number']

        def abs_path(path):
            return os.path.join(self.args['base_dir'], f'assets/track_{track_num}', path)

        self.car = Car(0, self.args['start_x'][track_num-1], self.args['start_y'][track_num-1], car_file=abs_path(self.args['car_file']), fps=self.args['fps'])
        self.road = Road(
            road_file=abs_path(self.args['road_file']),
            center_lane_file=abs_path(self.args['center_lane_file']),
            textures=self.args['textures'],
            texture_files=[abs_path(f) for f in self.args['texture_files']],
            texture_frictions=self.args['texture_frictions'],
            use_textures=self.args['use_textures']
        )
        self.road.blit(self.legacy_screen)
        self.car.blit(self.legacy_screen, [0, 0])
        return self.obs

    def render(self, mode='human'):
        pg.display.update()

    def close(self):
        pg.quit()