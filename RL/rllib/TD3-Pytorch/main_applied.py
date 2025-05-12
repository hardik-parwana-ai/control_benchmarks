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




# EnvName:AC1, Steps: 2k, Episode Reward:-16649
# EnvName:AC1, Steps: 4k, Episode Reward:-15348
# EnvName:AC1, Steps: 6k, Episode Reward:-14961
# EnvName:AC1, Steps: 8k, Episode Reward:-14501
# EnvName:AC1, Steps: 10k, Episode Reward:-16665
# EnvName:AC1, Steps: 12k, Episode Reward:-15233
# EnvName:AC1, Steps: 14k, Episode Reward:-13701
# EnvName:AC1, Steps: 16k, Episode Reward:-10549
# EnvName:AC1, Steps: 18k, Episode Reward:-14275
# EnvName:AC1, Steps: 20k, Episode Reward:-14369
# EnvName:AC1, Steps: 22k, Episode Reward:-9278
# EnvName:AC1, Steps: 24k, Episode Reward:-10751
# EnvName:AC1, Steps: 26k, Episode Reward:-9614
# EnvName:AC1, Steps: 28k, Episode Reward:-6443
# EnvName:AC1, Steps: 30k, Episode Reward:-11203
# EnvName:AC1, Steps: 32k, Episode Reward:-7042
# EnvName:AC1, Steps: 34k, Episode Reward:-6499
# EnvName:AC1, Steps: 36k, Episode Reward:-6334
# EnvName:AC1, Steps: 38k, Episode Reward:-6781
# EnvName:AC1, Steps: 40k, Episode Reward:-4758
# EnvName:AC1, Steps: 42k, Episode Reward:-5876
# EnvName:AC1, Steps: 44k, Episode Reward:-4653
# EnvName:AC1, Steps: 46k, Episode Reward:-4079
# EnvName:AC1, Steps: 48k, Episode Reward:-3568
# EnvName:AC1, Steps: 50k, Episode Reward:-3670
# EnvName:AC1, Steps: 52k, Episode Reward:-3675
# EnvName:AC1, Steps: 54k, Episode Reward:-3369
# EnvName:AC1, Steps: 56k, Episode Reward:-9221
# EnvName:AC1, Steps: 58k, Episode Reward:-8913
# EnvName:AC1, Steps: 60k, Episode Reward:-5479
# EnvName:AC1, Steps: 62k, Episode Reward:-3728
# EnvName:AC1, Steps: 64k, Episode Reward:-3829
# EnvName:AC1, Steps: 66k, Episode Reward:-3132
# EnvName:AC1, Steps: 68k, Episode Reward:-4141
# EnvName:AC1, Steps: 70k, Episode Reward:-4013
# EnvName:AC1, Steps: 72k, Episode Reward:-10622
# EnvName:AC1, Steps: 74k, Episode Reward:-4869
# EnvName:AC1, Steps: 76k, Episode Reward:-3907
# EnvName:AC1, Steps: 78k, Episode Reward:-4207
# EnvName:AC1, Steps: 80k, Episode Reward:-3441
# EnvName:AC1, Steps: 82k, Episode Reward:-3130
# EnvName:AC1, Steps: 84k, Episode Reward:-3445
# EnvName:AC1, Steps: 86k, Episode Reward:-6563
# EnvName:AC1, Steps: 88k, Episode Reward:-9993
# EnvName:AC1, Steps: 90k, Episode Reward:-4062
# EnvName:AC1, Steps: 92k, Episode Reward:-10379
# EnvName:AC1, Steps: 94k, Episode Reward:-5912
# EnvName:AC1, Steps: 96k, Episode Reward:-9217
# EnvName:AC1, Steps: 98k, Episode Reward:-3999
# EnvName:AC1, Steps: 100k, Episode Reward:-3115
# EnvName:AC1, Steps: 102k, Episode Reward:-3974
# EnvName:AC1, Steps: 104k, Episode Reward:-3312
# EnvName:AC1, Steps: 106k, Episode Reward:-3975
# EnvName:AC1, Steps: 108k, Episode Reward:-2925
# EnvName:AC1, Steps: 110k, Episode Reward:-3285
# EnvName:AC1, Steps: 112k, Episode Reward:-3960
# EnvName:AC1, Steps: 114k, Episode Reward:-2895
# EnvName:AC1, Steps: 116k, Episode Reward:-3199
# EnvName:AC1, Steps: 118k, Episode Reward:-3082
# EnvName:AC1, Steps: 120k, Episode Reward:-3581
# EnvName:AC1, Steps: 122k, Episode Reward:-10469
# EnvName:AC1, Steps: 124k, Episode Reward:-3180
# EnvName:AC1, Steps: 126k, Episode Reward:-5194
# EnvName:AC1, Steps: 128k, Episode Reward:-10583
# EnvName:AC1, Steps: 130k, Episode Reward:-3529
# EnvName:AC1, Steps: 132k, Episode Reward:-3143
# EnvName:AC1, Steps: 134k, Episode Reward:-3434
# EnvName:AC1, Steps: 136k, Episode Reward:-2811
# EnvName:AC1, Steps: 138k, Episode Reward:-3645
# EnvName:AC1, Steps: 140k, Episode Reward:-3258
# EnvName:AC1, Steps: 142k, Episode Reward:-2732
# EnvName:AC1, Steps: 144k, Episode Reward:-4951
# EnvName:AC1, Steps: 146k, Episode Reward:-10636
# EnvName:AC1, Steps: 148k, Episode Reward:-9147
# EnvName:AC1, Steps: 150k, Episode Reward:-3272
# EnvName:AC1, Steps: 152k, Episode Reward:-2924
# EnvName:AC1, Steps: 154k, Episode Reward:-4405
# EnvName:AC1, Steps: 156k, Episode Reward:-3634
# EnvName:AC1, Steps: 158k, Episode Reward:-8479
# EnvName:AC1, Steps: 160k, Episode Reward:-11258
# EnvName:AC1, Steps: 162k, Episode Reward:-3386
# EnvName:AC1, Steps: 164k, Episode Reward:-4068
# EnvName:AC1, Steps: 166k, Episode Reward:-12194
# EnvName:AC1, Steps: 168k, Episode Reward:-4034
# EnvName:AC1, Steps: 170k, Episode Reward:-9940
# EnvName:AC1, Steps: 172k, Episode Reward:-3322
# EnvName:AC1, Steps: 174k, Episode Reward:-3388
# EnvName:AC1, Steps: 176k, Episode Reward:-10945
# EnvName:AC1, Steps: 178k, Episode Reward:-3490
# EnvName:AC1, Steps: 180k, Episode Reward:-4077
# EnvName:AC1, Steps: 182k, Episode Reward:-3646
# EnvName:AC1, Steps: 184k, Episode Reward:-9632
# EnvName:AC1, Steps: 186k, Episode Reward:-3523
# EnvName:AC1, Steps: 188k, Episode Reward:-4180
# EnvName:AC1, Steps: 190k, Episode Reward:-10160
# EnvName:AC1, Steps: 192k, Episode Reward:-3937
# EnvName:AC1, Steps: 194k, Episode Reward:-2596
# EnvName:AC1, Steps: 196k, Episode Reward:-3550
# EnvName:AC1, Steps: 198k, Episode Reward:-4610
# EnvName:AC1, Steps: 200k, Episode Reward:-4415
# EnvName:AC1, Steps: 202k, Episode Reward:-3746
# EnvName:AC1, Steps: 204k, Episode Reward:-3535
# EnvName:AC1, Steps: 206k, Episode Reward:-9046
# EnvName:AC1, Steps: 208k, Episode Reward:-4087
# EnvName:AC1, Steps: 210k, Episode Reward:-11416
# EnvName:AC1, Steps: 212k, Episode Reward:-11185
# EnvName:AC1, Steps: 214k, Episode Reward:-4189
# EnvName:AC1, Steps: 216k, Episode Reward:-2819
# EnvName:AC1, Steps: 218k, Episode Reward:-7656
# EnvName:AC1, Steps: 220k, Episode Reward:-3440
# EnvName:AC1, Steps: 222k, Episode Reward:-3407
# EnvName:AC1, Steps: 224k, Episode Reward:-11545
# EnvName:AC1, Steps: 226k, Episode Reward:-3183
# EnvName:AC1, Steps: 228k, Episode Reward:-3863
# EnvName:AC1, Steps: 230k, Episode Reward:-7452
# EnvName:AC1, Steps: 232k, Episode Reward:-12274
# EnvName:AC1, Steps: 234k, Episode Reward:-3996
# EnvName:AC1, Steps: 236k, Episode Reward:-5464
# EnvName:AC1, Steps: 238k, Episode Reward:-2919
# EnvName:AC1, Steps: 240k, Episode Reward:-3566
# EnvName:AC1, Steps: 242k, Episode Reward:-4322
# EnvName:AC1, Steps: 244k, Episode Reward:-9018
# EnvName:AC1, Steps: 246k, Episode Reward:-3334
# EnvName:AC1, Steps: 248k, Episode Reward:-4607
# EnvName:AC1, Steps: 250k, Episode Reward:-3561
# EnvName:AC1, Steps: 252k, Episode Reward:-11074
# EnvName:AC1, Steps: 254k, Episode Reward:-10885
# EnvName:AC1, Steps: 256k, Episode Reward:-3646
# EnvName:AC1, Steps: 258k, Episode Reward:-3502
# EnvName:AC1, Steps: 260k, Episode Reward:-7522
# EnvName:AC1, Steps: 262k, Episode Reward:-3623
# EnvName:AC1, Steps: 264k, Episode Reward:-4377
# EnvName:AC1, Steps: 266k, Episode Reward:-7447
# EnvName:AC1, Steps: 268k, Episode Reward:-3430
# EnvName:AC1, Steps: 270k, Episode Reward:-11407
# EnvName:AC1, Steps: 272k, Episode Reward:-3367
# EnvName:AC1, Steps: 274k, Episode Reward:-10665
# EnvName:AC1, Steps: 276k, Episode Reward:-3722
# EnvName:AC1, Steps: 278k, Episode Reward:-10996
# EnvName:AC1, Steps: 280k, Episode Reward:-4366
# EnvName:AC1, Steps: 282k, Episode Reward:-3596
# EnvName:AC1, Steps: 284k, Episode Reward:-3830
# EnvName:AC1, Steps: 286k, Episode Reward:-4327
# EnvName:AC1, Steps: 288k, Episode Reward:-12429
# EnvName:AC1, Steps: 290k, Episode Reward:-2887
# EnvName:AC1, Steps: 292k, Episode Reward:-3410
# EnvName:AC1, Steps: 294k, Episode Reward:-3519
# EnvName:AC1, Steps: 296k, Episode Reward:-7040
# EnvName:AC1, Steps: 298k, Episode Reward:-9613
# EnvName:AC1, Steps: 300k, Episode Reward:-4612
# EnvName:AC1, Steps: 302k, Episode Reward:-5571
# EnvName:AC1, Steps: 304k, Episode Reward:-4972
# EnvName:AC1, Steps: 306k, Episode Reward:-9725
# EnvName:AC1, Steps: 308k, Episode Reward:-4433
# EnvName:AC1, Steps: 310k, Episode Reward:-3499
# EnvName:AC1, Steps: 312k, Episode Reward:-10429
# EnvName:AC1, Steps: 314k, Episode Reward:-13059
# EnvName:AC1, Steps: 316k, Episode Reward:-9317
# EnvName:AC1, Steps: 318k, Episode Reward:-3919
# EnvName:AC1, Steps: 320k, Episode Reward:-4026
# EnvName:AC1, Steps: 322k, Episode Reward:-4050
# EnvName:AC1, Steps: 324k, Episode Reward:-3337
# EnvName:AC1, Steps: 326k, Episode Reward:-5267
# EnvName:AC1, Steps: 328k, Episode Reward:-3929
# EnvName:AC1, Steps: 330k, Episode Reward:-4314
# EnvName:AC1, Steps: 332k, Episode Reward:-4375
# EnvName:AC1, Steps: 334k, Episode Reward:-12122
# EnvName:AC1, Steps: 336k, Episode Reward:-3081
# EnvName:AC1, Steps: 338k, Episode Reward:-5238
# EnvName:AC1, Steps: 340k, Episode Reward:-12873
# EnvName:AC1, Steps: 342k, Episode Reward:-3082
# EnvName:AC1, Steps: 344k, Episode Reward:-4825
# EnvName:AC1, Steps: 346k, Episode Reward:-4723
# EnvName:AC1, Steps: 348k, Episode Reward:-3383
# EnvName:AC1, Steps: 350k, Episode Reward:-4001
# EnvName:AC1, Steps: 352k, Episode Reward:-4506
# EnvName:AC1, Steps: 354k, Episode Reward:-4516
# EnvName:AC1, Steps: 356k, Episode Reward:-3488
# EnvName:AC1, Steps: 358k, Episode Reward:-10771
# EnvName:AC1, Steps: 360k, Episode Reward:-8863
# EnvName:AC1, Steps: 362k, Episode Reward:-5313
# EnvName:AC1, Steps: 364k, Episode Reward:-4711
# EnvName:AC1, Steps: 366k, Episode Reward:-3260
# EnvName:AC1, Steps: 368k, Episode Reward:-4457
# EnvName:AC1, Steps: 370k, Episode Reward:-12403
# EnvName:AC1, Steps: 372k, Episode Reward:-3834
# EnvName:AC1, Steps: 374k, Episode Reward:-12589
# EnvName:AC1, Steps: 376k, Episode Reward:-3136
# EnvName:AC1, Steps: 378k, Episode Reward:-11919
# EnvName:AC1, Steps: 380k, Episode Reward:-11782
# EnvName:AC1, Steps: 382k, Episode Reward:-3485
# EnvName:AC1, Steps: 384k, Episode Reward:-12920
# EnvName:AC1, Steps: 386k, Episode Reward:-6357
# EnvName:AC1, Steps: 388k, Episode Reward:-6403
# EnvName:AC1, Steps: 390k, Episode Reward:-12638
# EnvName:AC1, Steps: 392k, Episode Reward:-6086
# EnvName:AC1, Steps: 394k, Episode Reward:-4644
# EnvName:AC1, Steps: 396k, Episode Reward:-5113
# EnvName:AC1, Steps: 398k, Episode Reward:-12837
# EnvName:AC1, Steps: 400k, Episode Reward:-4650
# EnvName:AC1, Steps: 402k, Episode Reward:-3545
# EnvName:AC1, Steps: 404k, Episode Reward:-4496
# EnvName:AC1, Steps: 406k, Episode Reward:-3401
# EnvName:AC1, Steps: 408k, Episode Reward:-4615
# EnvName:AC1, Steps: 410k, Episode Reward:-3314
# EnvName:AC1, Steps: 412k, Episode Reward:-2963
# EnvName:AC1, Steps: 414k, Episode Reward:-5166
# EnvName:AC1, Steps: 416k, Episode Reward:-4943
# EnvName:AC1, Steps: 418k, Episode Reward:-3941
# EnvName:AC1, Steps: 420k, Episode Reward:-4129
# EnvName:AC1, Steps: 422k, Episode Reward:-10201
# EnvName:AC1, Steps: 424k, Episode Reward:-12654
# EnvName:AC1, Steps: 426k, Episode Reward:-4149
# EnvName:AC1, Steps: 428k, Episode Reward:-3059
# EnvName:AC1, Steps: 430k, Episode Reward:-4663
# EnvName:AC1, Steps: 432k, Episode Reward:-10767
# EnvName:AC1, Steps: 434k, Episode Reward:-4086
# EnvName:AC1, Steps: 436k, Episode Reward:-2814
# EnvName:AC1, Steps: 438k, Episode Reward:-4379
# EnvName:AC1, Steps: 440k, Episode Reward:-12646
# EnvName:AC1, Steps: 442k, Episode Reward:-11836
# EnvName:AC1, Steps: 444k, Episode Reward:-3734
# EnvName:AC1, Steps: 446k, Episode Reward:-4446
# EnvName:AC1, Steps: 448k, Episode Reward:-4483
# EnvName:AC1, Steps: 450k, Episode Reward:-11595
# EnvName:AC1, Steps: 452k, Episode Reward:-7463
# EnvName:AC1, Steps: 454k, Episode Reward:-11471
# EnvName:AC1, Steps: 456k, Episode Reward:-4389
# EnvName:AC1, Steps: 458k, Episode Reward:-5283
# EnvName:AC1, Steps: 460k, Episode Reward:-4525
# EnvName:AC1, Steps: 462k, Episode Reward:-5124
# EnvName:AC1, Steps: 464k, Episode Reward:-4366
# EnvName:AC1, Steps: 466k, Episode Reward:-9979
# EnvName:AC1, Steps: 468k, Episode Reward:-4391
# EnvName:AC1, Steps: 470k, Episode Reward:-4844
# EnvName:AC1, Steps: 472k, Episode Reward:-6024
# EnvName:AC1, Steps: 474k, Episode Reward:-3782
# EnvName:AC1, Steps: 476k, Episode Reward:-4367
# EnvName:AC1, Steps: 478k, Episode Reward:-4206
# EnvName:AC1, Steps: 480k, Episode Reward:-9104
# EnvName:AC1, Steps: 482k, Episode Reward:-10923
# EnvName:AC1, Steps: 484k, Episode Reward:-3610
# EnvName:AC1, Steps: 486k, Episode Reward:-5110
# EnvName:AC1, Steps: 488k, Episode Reward:-4842
# EnvName:AC1, Steps: 490k, Episode Reward:-4434
# EnvName:AC1, Steps: 492k, Episode Reward:-3543
# EnvName:AC1, Steps: 494k, Episode Reward:-4170
# EnvName:AC1, Steps: 496k, Episode Reward:-3550
# EnvName:AC1, Steps: 498k, Episode Reward:-4329
# EnvName:AC1, Steps: 500k, Episode Reward:-10819
# EnvName:AC1, Steps: 502k, Episode Reward:-8519
# EnvName:AC1, Steps: 504k, Episode Reward:-3734
# EnvName:AC1, Steps: 506k, Episode Reward:-3299
# EnvName:AC1, Steps: 508k, Episode Reward:-4222
# EnvName:AC1, Steps: 510k, Episode Reward:-3142
# EnvName:AC1, Steps: 512k, Episode Reward:-3705
# EnvName:AC1, Steps: 514k, Episode Reward:-4902
# EnvName:AC1, Steps: 516k, Episode Reward:-3936
# EnvName:AC1, Steps: 518k, Episode Reward:-5034
# EnvName:AC1, Steps: 520k, Episode Reward:-4091
# EnvName:AC1, Steps: 522k, Episode Reward:-4394
# EnvName:AC1, Steps: 524k, Episode Reward:-4177
# EnvName:AC1, Steps: 526k, Episode Reward:-3916
# EnvName:AC1, Steps: 528k, Episode Reward:-10817
# EnvName:AC1, Steps: 530k, Episode Reward:-7437
# EnvName:AC1, Steps: 532k, Episode Reward:-11869
# EnvName:AC1, Steps: 534k, Episode Reward:-9122
# EnvName:AC1, Steps: 536k, Episode Reward:-5480
# EnvName:AC1, Steps: 538k, Episode Reward:-4202
# EnvName:AC1, Steps: 540k, Episode Reward:-5538
# EnvName:AC1, Steps: 542k, Episode Reward:-3775
# EnvName:AC1, Steps: 544k, Episode Reward:-5230
# EnvName:AC1, Steps: 546k, Episode Reward:-10140
# EnvName:AC1, Steps: 548k, Episode Reward:-2890
# EnvName:AC1, Steps: 550k, Episode Reward:-5909
# EnvName:AC1, Steps: 552k, Episode Reward:-3858
# EnvName:AC1, Steps: 554k, Episode Reward:-4337
# EnvName:AC1, Steps: 556k, Episode Reward:-5212
# EnvName:AC1, Steps: 558k, Episode Reward:-4966
# EnvName:AC1, Steps: 560k, Episode Reward:-4147
# EnvName:AC1, Steps: 562k, Episode Reward:-4417
# EnvName:AC1, Steps: 564k, Episode Reward:-9546
# EnvName:AC1, Steps: 566k, Episode Reward:-4971
# EnvName:AC1, Steps: 568k, Episode Reward:-11504
# EnvName:AC1, Steps: 570k, Episode Reward:-3710
# EnvName:AC1, Steps: 572k, Episode Reward:-3461
# EnvName:AC1, Steps: 574k, Episode Reward:-5724
# EnvName:AC1, Steps: 576k, Episode Reward:-3629
# EnvName:AC1, Steps: 578k, Episode Reward:-3090
# EnvName:AC1, Steps: 580k, Episode Reward:-3644
# EnvName:AC1, Steps: 582k, Episode Reward:-4404
# EnvName:AC1, Steps: 584k, Episode Reward:-3925
# EnvName:AC1, Steps: 586k, Episode Reward:-4827
# EnvName:AC1, Steps: 588k, Episode Reward:-4685
# EnvName:AC1, Steps: 590k, Episode Reward:-5586
# EnvName:AC1, Steps: 592k, Episode Reward:-4709
# EnvName:AC1, Steps: 594k, Episode Reward:-3200
# EnvName:AC1, Steps: 596k, Episode Reward:-4699
# EnvName:AC1, Steps: 598k, Episode Reward:-9002
# EnvName:AC1, Steps: 600k, Episode Reward:-4841
# EnvName:AC1, Steps: 602k, Episode Reward:-3870
# EnvName:AC1, Steps: 604k, Episode Reward:-11474
# EnvName:AC1, Steps: 606k, Episode Reward:-3329
# EnvName:AC1, Steps: 608k, Episode Reward:-12533
# EnvName:AC1, Steps: 610k, Episode Reward:-12811
# EnvName:AC1, Steps: 612k, Episode Reward:-4231
# EnvName:AC1, Steps: 614k, Episode Reward:-5551
# EnvName:AC1, Steps: 616k, Episode Reward:-12667
# EnvName:AC1, Steps: 618k, Episode Reward:-4293
# EnvName:AC1, Steps: 620k, Episode Reward:-3583
# EnvName:AC1, Steps: 622k, Episode Reward:-6012
# EnvName:AC1, Steps: 624k, Episode Reward:-4713
# EnvName:AC1, Steps: 626k, Episode Reward:-4894
# EnvName:AC1, Steps: 628k, Episode Reward:-9056
# EnvName:AC1, Steps: 630k, Episode Reward:-4368
# EnvName:AC1, Steps: 632k, Episode Reward:-13073
# EnvName:AC1, Steps: 634k, Episode Reward:-4348
# EnvName:AC1, Steps: 636k, Episode Reward:-12320
# EnvName:AC1, Steps: 638k, Episode Reward:-4757
# EnvName:AC1, Steps: 640k, Episode Reward:-11561
# EnvName:AC1, Steps: 642k, Episode Reward:-10419
# EnvName:AC1, Steps: 644k, Episode Reward:-10497
# EnvName:AC1, Steps: 646k, Episode Reward:-4640
# EnvName:AC1, Steps: 648k, Episode Reward:-3660
# EnvName:AC1, Steps: 650k, Episode Reward:-3182
# EnvName:AC1, Steps: 652k, Episode Reward:-7790
# EnvName:AC1, Steps: 654k, Episode Reward:-4289
# EnvName:AC1, Steps: 656k, Episode Reward:-9803
# EnvName:AC1, Steps: 658k, Episode Reward:-7413
# EnvName:AC1, Steps: 660k, Episode Reward:-7953
# EnvName:AC1, Steps: 662k, Episode Reward:-7145
# EnvName:AC1, Steps: 664k, Episode Reward:-4995
# EnvName:AC1, Steps: 666k, Episode Reward:-3999
# EnvName:AC1, Steps: 668k, Episode Reward:-3250
# EnvName:AC1, Steps: 670k, Episode Reward:-5021
# EnvName:AC1, Steps: 672k, Episode Reward:-5183
# EnvName:AC1, Steps: 674k, Episode Reward:-10499
# EnvName:AC1, Steps: 676k, Episode Reward:-3573
# EnvName:AC1, Steps: 678k, Episode Reward:-3427
# EnvName:AC1, Steps: 680k, Episode Reward:-3126
# EnvName:AC1, Steps: 682k, Episode Reward:-4801
# EnvName:AC1, Steps: 684k, Episode Reward:-7908
# EnvName:AC1, Steps: 686k, Episode Reward:-3989
# EnvName:AC1, Steps: 688k, Episode Reward:-4039
# EnvName:AC1, Steps: 690k, Episode Reward:-4240
# EnvName:AC1, Steps: 692k, Episode Reward:-4888
# EnvName:AC1, Steps: 694k, Episode Reward:-4855
# EnvName:AC1, Steps: 696k, Episode Reward:-11666
# EnvName:AC1, Steps: 698k, Episode Reward:-15353
# EnvName:AC1, Steps: 700k, Episode Reward:-12269
# python3 main_applied.py --EnvIdex 6 --render True --Loadmodel True --ModelIdex 436
