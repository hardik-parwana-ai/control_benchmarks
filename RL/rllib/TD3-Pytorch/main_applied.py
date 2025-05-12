from utils import str2bool,evaluate_policy, Reward_adapter, evaluate_policy_car
from datetime import datetime
from TD3 import TD3_agent
import gymnasium as gym
import numpy as np
import os, shutil
import argparse
import torch
import pdb
from rewards import compute_reward
from vehicle_models import animate_trajectory


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=25, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(700000), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.95, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-2, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-2, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.35, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class CarWorldEnv(gym.Env):
    # Write the constructor and provide a single `config` arg,
    # which may be set to None by default.
    def __init__(self, config=None):
        # As per gymnasium standard, provide observation and action spaces in your
        # constructor.

        # state: X, Y, PSI, DELTA, V
        # Observation: X, Y, PSI, DELTA, V
        self.dt = 0.05
        self.init_state = [0.0,0.0,0.0,0.0,0.0] #np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        self.time = 0
        self.final_time = 20
        self.total_reward = 0
        self._max_episode_steps = int(self.final_time/self.dt)
        self.observation_space = gym.spaces.Box(
                                            np.array([-1000, -1000, -3.5, -3.5, -10000]),
                                            np.array([ 1000,   1000, 3.5, 3.5, 10000]),
                                            (5,), np.float32)
        self.action_space = gym.spaces.Box(np.array([0,-1]),
                                           np.array([1, 1]),                                           
                                           (2,),
                                           dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        # Return (reset) observation and info dict.
        self.state = np.copy(self.init_state)
        self.time = 0
        self.total_reward = 0
        return self.state, {}
    
    # def compute_reward( state, action ):
    #     print(f"hello from wrong")
    #     return state[0] * state[0]

    def step(self, action):
        # print(f"action: {action}")
        # Return next observation, reward, terminated, truncated, and info dict.
        X = np.array([ self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] ])
        U = np.array([action[0], action[1]]) #, action[2]])
        # action[1] = action[1] / 20
        # Now clip: steering between -80 and 80 degrees
        steering_limit = 40.0 * np.pi / 180
        if np.abs(X[3] + action[1] * self.dt) > steering_limit:
            action[1] = (np.sign(X[3] + action[1] * self.dt) * steering_limit - X[3]) / self.dt
        new_steering = X[3] + action[1] * self.dt
        # if new_steering > 
        speed_limit = 45.0
        if np.abs(X[4] + action[0] * self.dt) > speed_limit:
            action[0] = (np.sign(X[4] + action[0] * self.dt) * speed_limit - X[4]) / self.dt
        # yaw_limit = np.pi/30
        # if np.abs(X[2] + X[3] * self.dt) > yaw_limit:

        wheel_base = 2.0
        X_dot = np.array([
            X[4]*np.cos(X[2]),
            X[4]*np.sin(X[2]),
            X[4] * np.tan(X[3]) / wheel_base,
            action[1],
            action[0]
        ])

        X_next = X + X_dot * self.dt
        X_next[3] = wrap_angle(X_next[3])
        X_next[4] = wrap_angle(X_next[4])

        self.state = np.copy(X_next)
        # self.state = [X_next[0], X_next[1], X_next[2], X_next[3], X_next[4]]

        reward = compute_reward( self.time, self.state, action )
        self.total_reward += reward
        self.time = self.time + self.dt
        # print(f"reward: {reward}, action: {action}, state: {self.state}")
        terminated = True if self.time > self.final_time else False
        info = {"episode": {"r": self.total_reward, "l": self.time}} if terminated else {}
        # print(f"total: reward: {self.total_reward}, info: {info}")
        return self.state, reward, terminated, False, info
    
    def render(self):
        pass

    def close(self):
        pass


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v3','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3', 'AppliedCar-v1']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3', 'AC1']

    # Build Env
    if opt.EnvIdex<6:        
        env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
        eval_env = gym.make(EnvName[opt.EnvIdex])
    else:
        env = CarWorldEnv() #gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
        eval_env = CarWorldEnv() #gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:AppliedCar  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = TD3_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        if opt.EnvIdex<6:
            score = evaluate_policy(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
        else:
            score, states = evaluate_policy_car(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
            animate_trajectory(states)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                if total_steps < (10*opt.max_e_steps): a = env.action_space.sample() # warm up
                else: a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                r = Reward_adapter(r, opt.EnvIdex)
                done = (dw or tr)
                # pdb.set_trace()
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for j in range(opt.update_every):
                        agent.train()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    agent.explore_noise *= opt.explore_noise_decay
                    ep_r = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()




