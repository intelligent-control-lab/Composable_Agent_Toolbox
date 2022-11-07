import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../'))
import numpy as np
import evaluator, agent, env
import time, yaml
import progressbar
from PIL import Image
import math

import copy, argparse
import os, torch, random, json
from torch import nn
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt
from argparse import Namespace
import shutil

# ---------------------------------------------------------------------------- #
#                                 action space                                 #
# ---------------------------------------------------------------------------- #
LIN_DISCRETE_SIZE = 5
LIN_MIN = -0.01
LIN_MAX = 0.01
ROT_DISCRETE_SIZE = 5
ROT_MIN = -10.0
ROT_MAX = 10.0
ACTION_SPACE_SIZE = LIN_DISCRETE_SIZE * ROT_DISCRETE_SIZE

def get_action(action_id):

    lin_id = action_id // ROT_DISCRETE_SIZE
    rot_id = action_id % ROT_DISCRETE_SIZE

    action = [
        LIN_MIN + (LIN_MAX-LIN_MIN)/(LIN_DISCRETE_SIZE-1) * lin_id,
        ROT_MIN + (ROT_MAX-ROT_MIN)/(ROT_DISCRETE_SIZE-1) * rot_id
    ]
    
    return np.asarray(action).reshape(-1, 1)

# ---------------------------------------------------------------------------- #
#                                  state space                                 #
# ---------------------------------------------------------------------------- #
STATE_DIMENSION = 12 # change this according to get_state()

# ---------------------------------------------------------------------------- #
#                                      RL                                      #
# ---------------------------------------------------------------------------- #

class QNetwork(nn.Module):

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env_name, state_dim, num_of_actions, hidden_size):
        super(QNetwork, self).__init__()

        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        self.env_name = env_name
        self.dS = state_dim
        self.nA = num_of_actions

        self.model = nn.Sequential(
            nn.Linear(self.dS, hidden_size),
            # nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.nA)
        )

    def forward(self, states):
        q_values = self.model(states) # [B, nA]

        return q_values

    def save_model_weights(self, out_path, suffix):
        # Helper function to save your model / weights.
        torch.save(self.state_dict(), os.path.join(out_path, "{}_{}.pt".format(self.env_name, suffix)))

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location=torch.device('cpu')
        self.load_state_dict(torch.load(weight_file, map_location=map_location))


class Replay_Memory():

    def __init__(self, memory_size=50000, ep_burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.buf = deque(maxlen=memory_size)
        self.ep_burn_in = ep_burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        '''
            transition := [s, a, r, s_next, done]

            returns list of transitions
        '''
        assert(len(self.buf) >= batch_size)
        return random.sample(self.buf, batch_size)

    def append(self, transition):
        # Appends transition to the memory.
        '''
            transition := [s, a, r, s_next, done]
        '''
        self.buf.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, args, env_spec, observation_dim, num_of_actions):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.args = args

        # Use GPU if there is one
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.Q = QNetwork(env_name=args.env_name, state_dim=STATE_DIMENSION, num_of_actions=ACTION_SPACE_SIZE, hidden_size=args.hidden_size).to(self.device)
        self.Qhat = QNetwork(env_name=args.env_name, state_dim=STATE_DIMENSION, num_of_actions=ACTION_SPACE_SIZE, hidden_size=args.hidden_size).to(self.device)
        # Copy the weights from Q to target Q
        self.Qhat.load_state_dict(self.Q.state_dict())
        # Optimizer for Q only
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr)

        self.env_spec = env_spec
        self.env = None
        self.step_env_instance = 0

        self.dS = observation_dim
        self.nA = num_of_actions

        # Create memory and initialize with transitions from random policy
        if not self.args.demo:
            self.memory = Replay_Memory(memory_size=50000, ep_burn_in=20)
            self.burn_in_memory()

        self.test_return = []
        self.test_return_at_ep = []
        self.train_td = []
        self.train_return = []
        self.train_tdG_at_ep = []
        self.count_train_solve = 0

        self.solved = False
        self.step = -1
        self.ep = -1

    def epsilon_greedy_policy(self, q_values, e):
        # Creating epsilon greedy probabilities to sample from.
        '''
            q_values [B, nA], tensor
            returns [B, nA], tensor
        '''
        B = q_values.shape[0]
        amax = q_values.argmax(1) # [B,]
        policy = torch.ones_like(q_values, device=q_values.device) * (e / self.nA) # [B, nA]
        policy[torch.arange(B, device=q_values.device), amax] += (1-e)
        policy = policy / torch.sum(policy, dim=1, keepdim=True)

        return policy

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        '''
            q_values [B, nA], tensor
            returns [B, nA], tensor
        '''
        B = q_values.shape[0]
        amax = q_values.argmax(1) # [B,]
        policy = torch.zeros_like(q_values, device=q_values.device)
        policy[torch.arange(B, device=q_values.device), amax] = 1.0
        
        return policy

    def act(self, policies):
        '''
            policies [B, nA], tensor

            returns action [B,], list
        '''
        assert(len(policies.shape)==2)
        actions = []
        for policy in policies:
            roll = np.random.uniform()
            a = 0
            cdf = policy[a]
            while cdf <= roll and a<(self.nA-1):
                a += 1
                cdf += policy[a]
            actions.append(a)
        
        return actions

    def state_to_tensor(self, s):
        '''
            [1, dS]
        '''
        return torch.from_numpy(np.asarray(s)).unsqueeze(0).float().to(self.device)

    def loss(self, y, q):
        '''
            Mean squared TD error
        '''
        error = y-q
        # return error.square().mean()
        return error.clamp(min=-self.args.grad_norm, max=self.args.grad_norm).square().mean()

    def get_state(self, measurement_groups):

        state = []

        # robot
        v           = measurement_groups['robot']['cartesian_sensor']['vel'].reshape(-1)
        vnorm       = np.linalg.norm(v)
        heading     = measurement_groups['robot']['state_sensor']['state'][2, 0]
        heading_vec = np.array([math.cos(heading), math.sin(heading)])

        # robot - goal
        e              = measurement_groups['robot']['goal_sensor']['rel_pos'].reshape(-1)
        dist_goal      = np.linalg.norm(e)
        heading_target = math.atan2(e[1], e[0])

        # add goal info
        state += list([dist_goal])
        state += list([np.dot(e, heading_vec)])
        state += list([vnorm])
        state += list([vnorm*math.cos(heading_target - heading)])
        state += list([vnorm*math.sin(heading_target - heading)])
        state += list([math.atan2(math.sin(heading_target - heading), math.cos(heading_target - heading))])

        # add obs info
        for obs, measurement in measurement_groups['robot']['obstacle_sensor'].items():

            e_obs       = measurement['rel_pos'].reshape(-1)
            dist_obs    = np.linalg.norm(e_obs)
            heading_obs = math.atan2(e_obs[1], e_obs[0])

            state += list([dist_obs])
            state += list([np.dot(e_obs, heading_vec)])
            state += list([vnorm])
            state += list([vnorm*math.cos(heading_obs - heading)])
            state += list([vnorm*math.sin(heading_obs - heading)])
            state += list([math.atan2(math.sin(heading_obs - heading), math.cos(heading_obs - heading))])

        return state

    def get_reward(self, measurement_groups, dphi):
        r = 0

        # close to goal
        rel_pos_goal = measurement_groups['robot']['goal_sensor']['rel_pos'].reshape(-1)
        dist_goal = np.linalg.norm(rel_pos_goal)
        rel_vel_goal = measurement_groups['robot']['goal_sensor']['rel_vel'].reshape(-1)
        heading_target = math.atan2(rel_pos_goal[1], rel_pos_goal[0])
        heading = measurement_groups['robot']['state_sensor']['state'][2, 0]

        if self.prev_goal_dist is None:
            ddist = 0
        else:
            ddist = dist_goal - self.prev_goal_dist

        r -= ddist*100
        r -= np.linalg.norm(rel_vel_goal)*0.1
        # r -= abs(math.atan2(math.sin(heading_target - heading), math.cos(heading_target - heading)))

        # large reward for reaching (not penalizing distance far away)
        if dist_goal < 0.3:
            r += 1
        
        # if measurement_groups['robot']['safety_gym_done']:
        #     r += 500

        # penalize dphi
        r -= dphi

        # other reward components?
        # print('dist_goal {:>8.4f}, ddist {:>8.4f}, rel_angle {:>8.4f}, rel_vel_goal {:>8.4f}, dphi {:>8.4f}'.format(
        #     dist_goal, ddist, abs(math.atan2(math.sin(heading_target - heading), math.cos(heading_target - heading))),
        #     np.linalg.norm(rel_vel_goal), dphi))

        self.prev_goal_dist = dist_goal

        return r

    def new_env_instance(self):

        # make a copy of original spec
        env_spec = copy.deepcopy(self.env_spec)

        # --------------------------------- setup env -------------------------------- #
        # The module specs for agents, specifies which task, model, planner, controller, sensor to use.

        # create computational agents
        agents = []
        for agent_name, agent_spec_file in env_spec['agent_comp_spec'].items():
            with open(agent_spec_file, 'r') as infile:
                agent_module_spec = yaml.load(infile, Loader=yaml.SafeLoader)
                agents.append(agent.ModelBasedAgent(agent_module_spec))

        # create env or reset existing one
        if 'suite_name' in env_spec and env_spec['suite_name'] is not None:
            if self.env is None:
                self.env = env.SafetyGymEnv(env_spec, agents)
            else:
                self.env.reset()

            env_instance = self.env
        else:
            env_instance = env.FlatEnv(env_spec, agents)

        # reset variables
        self.prev_goal_dist = None
        self.step_env_instance = 0

        return env_instance, agents
    
    def env_step(self, env_instance, agents, measurement_groups, action, dt):

        self.step_env_instance += 1

        # agent routine
        actions = {}
        for agent in agents:
            # an action is dictionary which must contain a key "control"
            if agent.name == 'robot':

                # debug
                if measurement_groups['robot']['safety_gym_env'].done:
                    import ipdb; ipdb.set_trace()

                actions[agent.name] = agent.action(dt, measurement_groups[agent.name], get_action(action))
                dphi = actions[agent.name]["delta_phi"]
            else:
                # model-based action
                actions[agent.name] = agent.action(dt, measurement_groups[agent.name])

        _, env_info, measurement_groups_next, img = env_instance.step(actions, render=self.args.render)

        # RL compatible
        s_next = self.get_state(measurement_groups_next)
        r = self.get_reward(measurement_groups_next, dphi)
        done = (self.step_env_instance >= 500) or ('safety_gym_done' in measurement_groups_next['robot'] and measurement_groups_next['robot']['safety_gym_done'])

        return s_next, r, done, measurement_groups_next, img

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        
        ep_after_solved = 0

        while self.ep < self.args.max_ep:
            
            self.ep += 1
            
            # ----------------------------------- test ----------------------------------- #
            if self.ep % self.args.test_freq == 0:
                print('testing...')
                G_list = self.test(ep_to_run=self.args.test_eps)
                self.test_return.append(np.asarray(G_list).mean())
                self.test_return_at_ep.append(self.ep)

            # checkpt every 1k episodes
            if self.ep % self.args.checkpt_freq == 0:
                self.Q.save_model_weights(out_path=self.args.checkpt_path, suffix='{}'.format(self.ep))

            # ---------------------------------------------------------------------------- #
            #                                  new episode                                 #
            # ---------------------------------------------------------------------------- #

            # new env instance
            env_instance, agents = self.new_env_instance()
            dt, env_info, measurement_groups = env_instance.reset()
            step_env_instance = 0

            # -------------------------------- setup state ------------------------------- #
            s = self.get_state(measurement_groups)

            G = 0.0
            episode_td = []

            while True:
                
                self.step += 1

                # ----------------------------- update Q network ----------------------------- #

                if self.step % self.args.C == 0:
                    self.Qhat.load_state_dict(self.Q.state_dict())

                # -------------------------- control under e-greedy -------------------------- #

                q_vals = self.Q(self.state_to_tensor(s))
                epsilon = self.args.epsilon_max - min( self.step / (10**4), self.args.epsilon_max - self.args.epsilon_min )
                policies = self.epsilon_greedy_policy(q_vals, epsilon)
                a = self.act(policies=policies)[0]
                
                # ---------------------------------- execute --------------------------------- #
                # debug
                if env_instance.safety_gym_env.done:
                    import ipdb; ipdb.set_trace()

                s_next, r, done, measurement_groups, img = self.env_step(env_instance, agents, measurement_groups, a, dt)
                G += r
                
                # ------------------------------- put in memory ------------------------------ #

                self.memory.append([s, a, r, s_next, done])

                # ------------------------- sample batch for training ------------------------ #

                batch = self.memory.sample_batch(batch_size=self.args.batch_size) # list of [s, a, r, s_next, done]

                # ------------------------------ compute target ------------------------------ #
                r_batch = torch.from_numpy(np.asarray([t[2] for t in batch])).reshape(-1, 1).float().to(self.device) # [B, 1]
                s_next_batch = torch.from_numpy(np.asarray([t[3] for t in batch])).float().to(self.device) # [B, dS]
                # compute next state value from target Q network, detach to prevent backprop
                v_next_batch = self.Qhat(s_next_batch).detach().max(dim=1, keepdim=True)[0] # [B, 1]
                done_batch = torch.from_numpy(np.asarray([(not t[-1]) for t in batch], dtype=float)).reshape(-1, 1).float().to(self.device) # [B, 1]

                y_batch = r_batch + done_batch * self.args.gamma * v_next_batch # [B, 1]
                y_batch = y_batch.squeeze(-1) # [B,]

                # -------------------------------- optimize Q -------------------------------- #

                # evaluate (s, a) by Q network
                s_batch = torch.from_numpy(np.asarray([t[0] for t in batch])).float().to(self.device) # [B, dS]
                a_batch = torch.from_numpy(np.asarray([t[1] for t in batch])).long().to(self.device) # [B,]
                q_batch = self.Q(s_batch)
                qa_batch = q_batch[torch.arange(self.args.batch_size, device=self.device), a_batch] # [B,]

                loss = self.loss(y_batch, qa_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # batch average TD error at each step
                episode_td.append((y_batch-qa_batch).abs().mean().item())

                # ------------------------------- miscellaneous ------------------------------ #
                if done:
                    break
                s = s_next
            

            env_instance.close()

            # ---------------------------------------------------------------------------- #
            #                                 Episode ends                                 #
            # ---------------------------------------------------------------------------- #

            # mean TD error through episode
            self.train_td.append(np.asarray(episode_td).mean())
            self.train_return.append(G)
            self.train_tdG_at_ep.append(self.ep)

            # ------------------------------- Log to screen ------------------------------ #
            solve_str = 'SOLVED' if self.solved else 'UNSOLVED'
            if self.ep % self.args.log_freq == 0:
                print('[{}] STEP {:7d} EP {:7d} TRAIN_G {:7.3f} TRAIN_TD {:7.3f} TEST_G {:7.3f}'.format(
                    solve_str, self.step, self.ep,
                    G, self.train_td[-1], self.test_return[-1]))

        print('\nTraining completed.')
        # ---------------------------------------------------------------------------- #
        #                                 Training ends                                #
        # ---------------------------------------------------------------------------- #

    def post_training(self):
        
        self.Q.save_model_weights(out_path=self.args.checkpt_path, suffix='{}_trained'.format(self.ep))

        out_path = self.args.exp_out_path

        np.save(os.path.join(out_path, 'test_return'), self.test_return)
        np.save(os.path.join(out_path, 'test_return_at_ep'), self.test_return_at_ep)

        np.save(os.path.join(out_path, 'train_td'), self.train_td)
        np.save(os.path.join(out_path, 'train_return'), self.train_return)
        np.save(os.path.join(out_path, 'train_tdG_at_ep'), self.train_tdG_at_ep)

        self.plot_and_save()

    def plot_and_save(self):

        out_path = self.args.exp_out_path

        test_return       = np.load(os.path.join(out_path, 'test_return.npy'))
        test_return_at_ep = np.load(os.path.join(out_path, 'test_return_at_ep.npy'))

        train_td        = np.load(os.path.join(out_path, 'train_td.npy'))
        train_return    = np.load(os.path.join(out_path, 'train_return.npy'))
        train_tdG_at_ep = np.load(os.path.join(out_path, 'train_tdG_at_ep.npy'))

        fig, axx = plt.subplots(1, 1)
        axx.plot(test_return_at_ep, test_return, label='Test return')
        axx.set_title("{} Test return through training".format(self.args.env_name))
        axx.set_xlabel("Episode")
        axx.set_ylabel("Return")
        fig.legend()
        plt.savefig(os.path.join(out_path, 'test_return_{}'.format(self.args.env_name)))

        # fig, axx = plt.subplots(1, 1)

        # ax_r = axx.twinx()
        # ax_r.tick_params(axis='y', labelcolor='r')
        # # ax_r.plot(train_tdG_at_ep, train_return, 'r:', linewidth=1, label='Train return')
        # ax_r.plot(test_return_at_ep, test_return, 'r', linewidth=1, label='Test return')
        # ax_r.set_ylabel("Return")

        # axx.plot(train_tdG_at_ep, train_td, label='TD error')
        # axx.set_title("{} TD error and test return through training".format(self.args.env_name))
        # axx.set_xlabel("Episode")
        # axx.set_ylabel("TD error")

        # fig.legend()
        # plt.savefig(os.path.join(out_path, 'TD_error_{}'.format(self.args.env_name)))

    def test(self, ep_to_run, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cummulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        Gs = []
        for _ in range(ep_to_run):

            G = 0.0

            # new env instance
            env_instance, agents = self.new_env_instance()
            dt, env_info, measurement_groups = env_instance.reset()

            # -------------------------------- setup state ------------------------------- #
            s = self.get_state(measurement_groups)

            while True:

                q_vals = self.Q(self.state_to_tensor(s)).detach()
                policies = self.greedy_policy(q_vals)
                a = self.act(policies=policies)[0]

                # ---------------------------------- execute --------------------------------- #
                if env_instance.safety_gym_env.done:
                    import ipdb; ipdb.set_trace()

                s_next, r, done, measurement_groups, img = self.env_step(env_instance, agents, measurement_groups, a, dt)
                
                G += r
                if done:
                    break
                s = s_next

            env_instance.close()
            Gs.append(G)

        return Gs

    def load_Q_weights_and_eval(self, weight_file):
        self.Q.load_model_weights(weight_file)
        self.Q.eval()

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        print('Burning in memory...')
        for _ in tqdm(range(self.memory.ep_burn_in)):

            # new env instance
            env_instance, agents = self.new_env_instance()
            dt, env_info, measurement_groups = env_instance.reset()

            # -------------------------------- setup state ------------------------------- #
            s = self.get_state(measurement_groups)

            while True:

                a = np.random.randint(ACTION_SPACE_SIZE)

                # ---------------------------------- execute --------------------------------- #

                s_next, r, done, measurement_groups, img = self.env_step(env_instance, agents, measurement_groups, a, dt)

                self.memory.append([s, a, r, s_next, done])
                if done:
                    break
                s = s_next

            env_instance.close()


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(rl_agent, exp, epi):

    save_path = os.path.join(rl_agent.args.exp_out_path, "video_ep_{}".format(epi))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    reward_total = []

    # new env instance
    env_instance, agents = rl_agent.new_env_instance()
    dt, env_info, measurement_groups = env_instance.reset()

    # -------------------------------- setup state ------------------------------- #
    state = rl_agent.get_state(measurement_groups)

    done = False
    while not done:

        q_vals = rl_agent.Q(rl_agent.state_to_tensor(state))
        policies = rl_agent.epsilon_greedy_policy(q_vals, 0.05)
        action = rl_agent.act(policies=policies)[0]

        next_state, reward, done, measurement_groups, img = rl_agent.env_step(env_instance, agents, measurement_groups, action, dt)

        state = next_state
        reward_total.append(reward)

        # save img
        if img is not None:
            im = Image.open(img)
            im.save(f'{save_path}/{rl_agent.step_env_instance:0>6d}.png')

    print("reward_total: {}".format(np.sum(reward_total)))
    env_instance.close()

    # make video
    os.system(f"ffmpeg -r 30 -f image2 -s 500x500 -i {save_path}/%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {save_path}/vid.mp4")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env_name', type=str)
    parser.add_argument('--max_ep', action="store", type=int)

    parser.add_argument('--hidden_size', action="store", type=int, default="64")
    parser.add_argument('--lr', action="store", type=float, default="1e-4")
    parser.add_argument('--grad_norm', action="store", type=float, default="100.0")
    parser.add_argument('--gamma', action="store", type=float, default="0.99")
    parser.add_argument('--batch_size', action="store", type=int, default="32")
    parser.add_argument('--epsilon_max', action="store", type=float, default="0.5")
    parser.add_argument('--epsilon_min', action="store", type=float, default="0.01")
    parser.add_argument('--C', action="store", type=int, default="400")
    parser.add_argument('--test_freq', action="store", type=int, default="50")
    parser.add_argument('--test_eps', action="store", type=int, default="20")
    parser.add_argument('--checkpt_freq', action="store", type=int, default="100")
    parser.add_argument('--log_freq', action="store", type=int, default="1")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--out_root', type=str, default='output')
    
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--demo_path', type=str, default='')
    parser.add_argument('--demo_ep', action="store", type=str, default="0")

    return parser.parse_args()

def main(args):

    # The environment specs, including specs for the phsical agent model,
    # physics engine scenario, rendering options, etc.
    with open('configs/safety_gym_env.yaml', 'r') as infile:
        env_spec = yaml.load(infile, Loader=yaml.SafeLoader)

    # --------------------------------- setup RL --------------------------------- #

    args = parse_arguments()

    if args.demo:
    
        with open(os.path.join(args.demo_path, "args.json"), "r") as infile:
            config = json.load(infile)
        args_saved = Namespace(**config)

        args_saved.demo = args.demo
        args_saved.demo_path = args.demo_path
        args_saved.demo_ep = args.demo_ep

        args = args_saved

    else:

        if not os.path.exists(args.out_root):
            os.mkdir(args.out_root)

        # key args
        args.exp_name = "{}_{}_C{}_H{}_Nm{}_lr{}".format(
            args.env_name, env_spec['world']['spec']['agent_goal_lists']['robot'],
            args.C, args.hidden_size, args.grad_norm, args.lr)

        # output root
        args.exp_out_path = os.path.join(args.out_root, args.exp_name)
        if os.path.exists(args.exp_out_path):
            shutil.rmtree(args.exp_out_path)
        os.mkdir(args.exp_out_path)

        # checkpt path
        args.checkpt_path = os.path.join(args.exp_out_path, 'checkpts')
        if not os.path.exists(args.checkpt_path):
            os.mkdir(args.checkpt_path)

        # tensor log path
        args.log_path = os.path.join(args.exp_out_path, 'log')

        # save args
        argparse_dict = vars(args)
        with open(os.path.join(args.exp_out_path, "args.json"), "w") as outfile:
            json.dump(argparse_dict, outfile, indent=4)

    argparse_dict = vars(args)
    arg_str = json.dumps(argparse_dict, indent=4)
    print(arg_str)

    # ------------------------------ create RL agent ----------------------------- #

    rl_agent = DQN_Agent(args, env_spec, STATE_DIMENSION, ACTION_SPACE_SIZE)
    
    if rl_agent.args.demo:
        checkpt_path = os.path.join(rl_agent.args.demo_path, 'checkpts', '{}_{}.pt'.format(rl_agent.args.env_name, rl_agent.args.demo_ep))
        rl_agent.load_Q_weights_and_eval(checkpt_path)
        test_video(rl_agent, rl_agent.args.exp_name, rl_agent.args.demo_ep)
        rl_agent.plot_and_save()
    else:
        print('Training started...')
        rl_agent.train()
        rl_agent.post_training()

if __name__ == '__main__':
    main(sys.argv)
