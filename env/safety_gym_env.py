from signal import pause
import numpy as np
import env.flat_world
import matplotlib.pyplot as plt
import importlib
import time, math
import copy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
from safety_gym.envs.engine import Engine




class SafetyGymEnv(object):
    def __init__(self, env_spec, comp_agents):
        '''
        Each environment has several pre-defined robot classes and sensor
        classes. The add_agent function will instantiate a robot class and 
        some sensors based on the specs.
        '''

        self.env_spec = copy.deepcopy(env_spec)
        self.dt = self.env_spec['dt']

        if self.env_spec['suite_name'] == 'mygoal1':

            # randomize but keep obstacle engaged

            env_config = {
                'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
                # 'robot_locations': [[robot_x, robot_y]],
                # 'robot_rot': 0,
                # 'goal_locations': [[goal_x, goal_y]],
                # 'hazards_locations': [[harzard_x, harzard_y]],
                'task': 'goal',
                'observation_flatten': True,  # Flatten observation into a vector
                'observe_sensors': True,  # Observe all sensor data from simulator
                # Sensor observations
                # Specify which sensors to add to observation space
                'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
                'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
                'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
                'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

                # observe goal/box/...
                'observe_goal_dist': False,  # Observe the distance to the goal
                'observe_goal_comp': False,  # Observe a compass vector to the goal
                'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
                'observe_box_comp': False,  # Observe the box with a compass
                'observe_box_lidar': False,  # Observe the box with a lidar
                'observe_circle': False,  # Observe the origin with a lidar
                'observe_remaining': False,  # Observe the fraction of steps remaining
                'observe_walls': False,  # Observe the walls with a lidar space
                'observe_hazards': True,  # Observe the vector from agent to hazards
                'observe_vases': True,  # Observe the vector from agent to vases
                'observe_pillars': False,  # Lidar observation of pillar object positions
                'observe_buttons': False,  # Lidar observation of button object positions
                'observe_gremlins': False,  # Gremlins are observed with lidar-like space
                'observe_vision': False,  # Observe vision from the robot

                # Constraints - flags which can be turned on
                # By default, no constraints are enabled, and all costs are indicator functions.
                'constrain_hazards': True,  # Constrain robot from being in hazardous areas
                'constrain_vases': False,  # Constrain frobot from touching objects
                'constrain_pillars': False,  # Immovable obstacles in the environment
                'constrain_buttons': False,  # Penalize pressing incorrect buttons
                'constrain_gremlins': False,  # Moving objects that must be avoided
                # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
                'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

                # lidar setting
                'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
                'lidar_num_bins': 16,
                # num setting
                'hazards_num': 1,
                'hazards_size': self.env_spec['hazards_size'],
                'vases_num': 0,

                # Frameskip is the number of physics simulation steps per environment step
                # Frameskip is sampled as a binomial distribution
                # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
                'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
                'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
            }
        else:
            raise NotImplementedError

        
        self.safety_gym_env = Engine(env_config)
        self.comp_agents = comp_agents
        self.reset()
        # setup rendering
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        # self.renderer = self.fig.canvas.renderer
        self.canvas = FigureCanvas(self.fig)

    def reset(self):
        o= self.safety_gym_env.reset()
        env_info = None
        sensor_data = dict()

        # -------------------------------- robot state ------------------------------- #
        sensor_data['robot'] = {}
        sensor_data['robot']['safety_gym_state'] = o
        sensor_data['robot']['safety_gym_reward'] = 0
        sensor_data['robot']['safety_gym_done'] = False
        sensor_data['robot']['safety_gym_cost_info'] = {'cost_hazards': 0, 'cost': 0}
        sensor_data['robot']['safety_gym_env'] = self.safety_gym_env

        robot_x = self.safety_gym_env.robot_pos[0]
        robot_y = self.safety_gym_env.robot_pos[1]

        # heading
        heading_vec = self.safety_gym_env.data.get_geom_xpos('pointarrow')[:2] - \
                        self.safety_gym_env.robot_pos[:2]
        robot_t = math.atan2(heading_vec[1], heading_vec[0])

        robot_velx = self.safety_gym_env.data.get_body_xvelp('robot')[0]
        robot_vely = self.safety_gym_env.data.get_body_xvelp('robot')[1]

        # heading vel
        robot_velt = self.safety_gym_env.data.get_body_xvelr('robot')[2]

        obstacle_x = self.safety_gym_env.hazards_pos[0][0]
        obstacle_y = self.safety_gym_env.hazards_pos[0][1]

        goal_x = self.safety_gym_env.goal_pos[0]
        goal_y = self.safety_gym_env.goal_pos[1]

        sensor_data['robot']['cartesian_sensor'] = {'pos': np.array([[robot_x], [robot_y]]), 'vel': np.array([[robot_velx], [robot_vely]])}
        sensor_data['robot']['state_sensor'] = {'state': np.array([[robot_x], [robot_y], [robot_t], [robot_velx], [robot_vely], [robot_velt]])}
        sensor_data['robot']['obstacle_sensor'] = {'hazard': {'rel_pos': np.array([[obstacle_x - robot_x], [obstacle_y - robot_y]]), 
                                                    'rel_vel': np.array([[0 - robot_velx], [0 - robot_vely]])} }
        sensor_data['robot']['goal_sensor'] = {'rel_pos': np.array([[goal_x - robot_x], [goal_y - robot_y]]), 
                                                    'rel_vel': np.array([[0 - robot_velx], [0 - robot_vely]]) }
        sensor_data['robot']['communication_sensor'] = {}


        return self.dt, env_info, sensor_data

    def step(self, actions, render=False, render_mode='human'):
        
        action_for_safety_gym = actions['robot']['control'].reshape(1, -1)

        o2, r, d, info = self.safety_gym_env.step(action_for_safety_gym)

        env_info = None
        sensor_data = dict()

        # -------------------------------- robot state ------------------------------- #
        sensor_data['robot'] = {}
        sensor_data['robot']['safety_gym_state'] = o2
        sensor_data['robot']['safety_gym_reward'] = r
        sensor_data['robot']['safety_gym_done'] = d
        sensor_data['robot']['safety_gym_cost_info'] = info # c = info.get('cost', 0)
        sensor_data['robot']['safety_gym_env'] = self.safety_gym_env

        robot_x = self.safety_gym_env.robot_pos[0]
        robot_y = self.safety_gym_env.robot_pos[1]

        # heading
        heading_vec = self.safety_gym_env.data.get_geom_xpos('pointarrow')[:2] - \
                        self.safety_gym_env.robot_pos[:2]
        robot_t = math.atan2(heading_vec[1], heading_vec[0])

        robot_velx = self.safety_gym_env.data.get_body_xvelp('robot')[0]
        robot_vely = self.safety_gym_env.data.get_body_xvelp('robot')[1]

        # heading vel
        robot_velt = self.safety_gym_env.data.get_body_xvelr('robot')[2]

        obstacle_x = self.safety_gym_env.hazards_pos[0][0]
        obstacle_y = self.safety_gym_env.hazards_pos[0][1]

        goal_x = self.safety_gym_env.goal_pos[0]
        goal_y = self.safety_gym_env.goal_pos[1]

        sensor_data['robot']['cartesian_sensor'] = {'pos': np.array([[robot_x], [robot_y]]), 'vel': np.array([[robot_velx], [robot_vely]])}
        sensor_data['robot']['state_sensor'] = {'state': np.array([[robot_x], [robot_y], [robot_t], [robot_velx], [robot_vely], [robot_velt]])}
        sensor_data['robot']['obstacle_sensor'] = {'hazard': {'rel_pos': np.array([[obstacle_x - robot_x], [obstacle_y - robot_y]]), 
                                                    'rel_vel': np.array([[0 - robot_velx], [0 - robot_vely]])} }
        sensor_data['robot']['goal_sensor'] = {'rel_pos': np.array([[goal_x - robot_x], [goal_y - robot_y]]), 
                                                    'rel_vel': np.array([[0 - robot_velx], [0 - robot_vely]]) }
        sensor_data['robot']['communication_sensor'] = {}

        # goal 
        #import ipdb; ipdb.set_trace()
        img = None
        if render:
            if render_mode == 'human':
                self.safety_gym_env.render()
                img = None
            elif render_mode == 'rgb':
                img = self.safety_gym_env.render(mode='rgb_array', width=1920, height=1200)

        return self.dt, env_info, sensor_data, img

    
    def close(self):
        plt.close(self.fig)

