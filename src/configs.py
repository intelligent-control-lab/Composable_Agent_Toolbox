import argparse

def add_experiment_args(parser):
    parser = argparse.ArgumentParser(description='DDPG for CarEnv')
    # ======================================
    # Experiment settings
    # ======================================
    exp = parser.add_argument_group("Reinforcement Learning Experiment Settings")
    exp.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='support option: train/test')
    exp.add_argument('--environment-id', default='ICLcar-v0', type=str, help='gym environment')
    exp.add_argument('--seed', default=1, type=int, help='seed')
    exp.add_argument('--num-envs', default=1, type=int, help='number of training environments')
    exp.add_argument('--log-frequency', default=5, type=int, help='number of episodes between log output')
    exp.add_argument('--fps', default=60, type=int, help='frequency between control updates in milliseconds')
    exp.add_argument('--render', default=False, type=bool, help='render')
    exp.add_argument('--exp-name', default='test', type=str, help='experiment name')
    exp.add_argument('--base-dir', default="", type=str, help='base directory running experiment from')

    exp.add_argument('--num-cpu', default=4, type=int, help='number of cpu for experiment grid')
    exp.add_argument('--use-gpu', default=True, type=bool, help='use gpu')
    exp.add_argument('-es', '--experiment-settings', default=["exp_settings/default_settings.yaml"], type=str, nargs='+', help='experiment settings yaml file')
    exp.add_argument('-eg', '--experiment-grid', default="", type=str, help='experiment grid yaml file')

    return parser

def add_logging_args(parser):
    # ======================================
    # Logging and model saving settings
    # ======================================
    log = parser.add_argument_group("Metadata logging Settings")
    log.add_argument('--log-dir', default='logs/', type=str, help='folder to store log files')
    log.add_argument('--save-frames', default=True, type=bool, help='save video frames')
    log.add_argument('--video-save-frequency', default=5, type=int, help='how many epochs between saving video')
    log.add_argument('--verbose', default=True, type=bool, help='output logging statements to terminal')


    return parser


def add_training_args(parser):
    # ======================================
    # Training settings
    # ======================================
    training = parser.add_argument_group("RL Training Settings")
    training.add_argument('--batch-size', default=64, type=int, help='batch size')
    training.add_argument('--buffer-size', default=1000000, type=int, help='replay buffer size')
    training.add_argument('--num-iterations', default=20000, type=int, help='total number of training steps')
    training.add_argument('--max-episode-length', default=200, type=int, help='maximum number of steps per episode')
    training.add_argument('--noise-stddev', default=0.2, type=float, help='standard deviation for ou noise')
    training.add_argument('--checkpoint-file', default='', type=str, help='file containing model checkpoint')

    return parser

def add_encoder_args(parser):
    # ======================================
    # rl agent settings
    # ======================================
    encoder = parser.add_argument_group("Image Encoder Settings")
    encoder.add_argument('--image-embedding-dimension', default=512, type=int, help='embedding dimension for image')
    encoder.add_argument('--context-embedding-dimension', default=512, type=int, help='embedding dimension for context information')

    return parser

def add_rl_agent_args(parser):
    # ======================================
    # rl agent settings
    # ======================================
    rl = parser.add_argument_group("Reinforcement Learning Agent Settings")
    rl.add_argument('--input-size', default=64, type=int, help='input size of image to model')
    rl.add_argument('--tau', default=0.001, type=float, help='update factor for soft update of target networks')
    rl.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    rl.add_argument('--actor-lr', default=1e-4, type=float, help='learning rate for actor network')
    rl.add_argument('--critic-lr', default=1e-3, type=float, help='learning rate for critic network')
    rl.add_argument('--uniform-init', default=[3e-4, 3e-3], type=float, nargs="+", help='initialize weight and bias of output layer in actor/critic networks')
    rl.add_argument('--hidden-sizes', default=[400, 300], type=int, nargs="+", help='hidden sizes for actor network')
    return parser

def add_car_env_args(parser):
    # ======================================
    # Car Env settings
    # ======================================
    env = parser.add_argument_group("Car Environment Settings")
    env.add_argument('--state-sources', default=['image', 'lane_direction', 'center_lane', 'lane_curvature', 'range_10'] , type=str, nargs='+', help='sources for the state space')
    env.add_argument('--crop-size', default=200, type=int, help='cropped bird eye image of car env')
    env.add_argument('--channels', default=1, type=int, help='number of channels in image input')
    env.add_argument('--env-mode', default='human', choices=['human', 'rgb'], type=str, help='env mode, currently supports: [human, rgb_array]')
    env.add_argument('--hide-display', default=False, type=bool, help='hide display for server training')
    env.add_argument('--use-textures', default=False, type=bool, help='include different road friction levels')
    env.add_argument('--act-limit', default=50, type=int, help='maximum value for left/right force')

    # reward function
    env.add_argument('--rotation-penalty-weight', default=0.008, type=float, help='weight for rotation penalty')
    env.add_argument('--distance-penalty-weight', default=0.012, type=float, help='weight for distance penalty')
    env.add_argument('--angle-penalty-weight', default=0.5, type=float, help='weight for angle difference penalty')
    env.add_argument('--velocity-reward-weight', default=0.01, type=float, help='weight for velocity reward')
    env.add_argument('--stationary-penalty-weight', default=1, type=float, help='weight for stationary penalty')

    # sprite files
    env.add_argument('--start-x', default=[740], type=int, nargs='+', help='start x')
    env.add_argument('--start-y', default=[240], type=int, nargs='+', help='start y')

    env.add_argument('--track-number', default=1, type=int, help='track file number')
    env.add_argument('--road-file', default='track_template.bmp', type=str, help='track file')
    env.add_argument('--car-file', default='car.bmp', type=str, help='car file')
    env.add_argument('--center-lane-file', default='center_lane.bmp', type=str, help='center lane file')
    env.add_argument('--textures', default=['icy', 'rocky'], nargs='+', type=str, help='types of road textures')
    env.add_argument('--texture-files', default=['icy.bmp', 'rocky.bmp'], nargs='+', type=str, help='road texture files')
    env.add_argument('--texture-frictions', default=[0.01, 3], nargs='+', type=int, help='road texture friction levels')

    return parser

def add_spinning_up_args(parser):
    spin_up = parser.add_argument_group("Spinning Up RL Algorithms Settings")
    spin_up.add_argument('--algo', default='ddpg', type=str, choices=['ddpg', 'sac'], help='which rl algorithm to use, currently supports: [ddpg, sac]')

    # ==============================
    # SHARED
    # ==============================
    spin_up.add_argument('--epochs', default=100, type=int, help='num epochs')
    spin_up.add_argument('--start-steps', default=10000, type=int, help='number of random action steps before real policy')
    spin_up.add_argument('--steps-per-epoch', default=4000, type=int, help='number of steps in each epoch')
    spin_up.add_argument('--update-after', default=1000, type=int, help='steps to collect before gradient updates')
    spin_up.add_argument('--update-every', default=500, type=int, help='steps between each update step')
    spin_up.add_argument('--num-test-episodes', default=10, type=int, help='number of test episodes')

    # ==============================
    # SAC - only
    # ==============================
    spin_up.add_argument('--learning-rate', default=1e-3, help='learning rate used for both policy and value function')
    spin_up.add_argument('--alpha', default=0.2, help='entropy regularization coefficient')

    return parser