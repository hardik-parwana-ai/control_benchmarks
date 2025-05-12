from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.td3 import PPOConfig
from ray import tune
import os
import gymnasium as gym
import numpy as np
from rewards import compute_reward
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback
import torch
from ray.rllib.core.rl_module import RLModule
from vehicle_models import animate_trajectory
# from ray.tune.logger import WandbLoggerCallback
from vehicle_models import animate_trajectory

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
        self.init_state = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float32)
        self.time = 0
        self.final_time = 50
        self.total_reward = 0
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
    #     print(f"wrong")
    #     return state[0] * state[0]

    def step(self, action):
        print(f"action: {action}")
        # Return next observation, reward, terminated, truncated, and info dict.
        X = np.array([ self.state[0], self.state[1], self.state[2], self.state[3], self.state[4] ])
        U = np.array([action[0], action[1]]) #, action[2]])
        action[1] = action[1] / 20
        # Now clip: steering between -80 and 80 degrees
        steering_limit = 45.0 * np.pi / 180
        if np.abs(X[3] + action[1] * self.dt) > steering_limit:
            action[1] = (np.sign(X[3] + action[1] * self.dt) * steering_limit - X[3]) / self.dt
        speed_limit = 20.0
        if np.abs(X[4] + action[0] * self.dt) > speed_limit:
            action[0] = (np.sign(X[4] + action[0] * self.dt) * speed_limit - X[4]) / self.dt

        wheel_base = 2.0
        X_dot = np.array([
            X[4]*np.cos(X[2]),
            X[4]*np.sin(X[2]),
            X[4] * np.tan(X[3]) / wheel_base,
            action[1],
            action[0]
        ], dtype=np.float32)

        X_next = X + X_dot * self.dt
        X_next[3] = wrap_angle(X_next[3])
        X_next[4] = wrap_angle(X_next[4])

        self.state = np.copy(X_next)

        reward = compute_reward( self.time, self.state, action )
        self.total_reward += reward
        self.time = self.time + self.dt
        
        terminated = True if self.time > self.final_time else False
        info = {"episode": {"r": self.total_reward, "l": self.time}} if terminated else {}
        # print(f"total: reward: {self.total_reward}, info: {info}")
        return self.state, reward, terminated, False, info
    
    def render(self):
        pass

    def close(self):
        pass


# class SaveCheckpointCallback(DefaultCallbacks):
#     def __init__(self, reward_threshold=200.0, N_save=5):
#         super().__init__()
#         self.reward_threshold = reward_threshold
#         self.N_save = N_save

#     def on_episode_end(self, *, algorithm, metrics_logger)

#     def on_train_result(self, *, algorithm, result, **kwargs):
#         # print(result)
#         reward_mean = result.get("episode_reward_mean", float("-inf"))
#         iteration = result.get("training_iteration", 0)

#         should_save = (
#             reward_mean >= self.reward_threshold or
#             iteration % self.N_save == 0
#         )

#         if should_save:
#             checkpoint_dir = os.path.join(os.getcwd(), f"ppo_checkpoints/{iteration}_checkpoint")
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             path = algorithm.save(checkpoint_dir)
#             print(f"✅ Checkpoint saved at iteration {iteration}, reward={reward_mean:.2f}, path={path}")
#         else:
#             print(f"ℹ️  Checkpoint skipped at iteration {iteration}, reward={reward_mean:.2f}")


import pdb
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    gym.register(
        id="gymnasium_env/CarWorldEnv-v0",
        entry_point=CarWorldEnv,
    )
    config = (
        PPOConfig()
        .environment(
            CarWorldEnv,
            env_config={},  # `config` to pass to your env class
        )
        # .environment("gymnasium_env/CarWorldEnv-v0",env_config={})
        .training(
            # train_batch_size=512,
            train_batch_size_per_learner=10000,
            num_epochs=50,
            lr=3e-4, #0.001, #2e-5,
            gamma=0.99,
            lambda_=0.95, #0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=0.5, #None,
            entropy_coeff=0.01, #0.1,
            vf_loss_coeff=0.5, #0.25,
            # sgd_minibatch_size=64,
            num_sgd_iter=100,
        )
        # .exploration(
        #     explore=True,
        #     entropy_coeff=0.01
        # )
        # .callbacks(SaveCheckpointCallback)
        .env_runners(num_env_runners=20) #, rollout_fragment_length=auto)
        .framework(framework="torch")
        # .debugging(log_level="ERROR")
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # algo = config.build()
    algo = config.build_algo()
    for _ in range(1):
        result = algo.train()
        print(f"next")
    # Run PPO with wandb logging
    # tune.run(
    #     "PPO",
    #     config=config.to_dict(),
    #     stop={"training_iteration": 20},
    #     callbacks=[
    #         WandbLoggerCallback(
    #             project="my-rayppo-project",
    #             name="ppo-run-1",
    #             group="ppo-experiments",
    #             log_config=True,
    #         )
    #     ],
    # )
        # reward_mean = result.get("episode_reward_mean", None)
        # print(f"Episode reward mean: {reward_mean if reward_mean is not None else 'N/A'}")
        # print(f"Episode reward mean: {result['episode_reward_mean']}")
        

    # Save final checkpoint manually
    checkpoint_dir = os.path.join(os.getcwd(), f"ppo_checkpoints/final_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # final_checkpoint = algo.save(checkpoint_dir)
    final_checkpoint = algo.save_to_path(checkpoint_dir)
    # print(f"Final checkpoint: {final_checkpoint}")

    # config.evaluation(
    #     # Run one evaluation round every iteration.
    #     evaluation_interval=1,

    #     # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
    #     evaluation_num_env_runners=2,

    #     # Run evaluation for exactly 10 episodes. Note that because you have
    #     # 2 EnvRunners, each one runs through 5 episodes.
    #     evaluation_duration_unit="episodes",
    #     evaluation_duration=10,
    # )

    # # Rebuild the PPO, but with the extra evaluation EnvRunnerGroup
    # ppo_with_evaluation = config.build_algo()
    # for _ in range(3):
    #     print(ppo_with_evaluation.train())

    ray.shutdown()

    # deploy model

    
    rl_module = RLModule.from_checkpoint(
        checkpoint_dir + "/learner_group" + "/learner" +  "/rl_module" + "/default_policy"
    )
    env = CarWorldEnv()
    episode_return = 0.0
    done = False

    # Reset the env to get the initial observation.
    obs, info = env.reset()

    car_states = env.state.reshape(-1,1)

    while not done:
        # Uncomment this line to render the env.
        # env.render()
        # Compute the next action from a batch (B=1) of observations.
        obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
        # print(f" ************************************ {obs_batch}")
        # model_outputs = rl_module.forward_inference({"obs": obs_batch})
        model_outputs = rl_module.forward_inference({"obs": obs_batch})
        # pdb.set_trace()
        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
        print(f"actions: {action_dist_params}")
        # We have continuous actions -> take the mean (max likelihood).
        greedy_action = np.clip(
            action_dist_params[0:2],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low,
            a_max=env.action_space.high,
        )
        # For discrete actions, you should take the argmax over the logits:
        # greedy_action = np.argmax(action_dist_params)

        # Send the action to the environment for the next step.
        obs, reward, terminated, truncated, info = env.step(greedy_action)
        car_states = np.append( car_states, obs.reshape(-1,1), axis=1 )

        # Perform env-loop bookkeeping.
        episode_return += reward
        done = terminated or truncated
    animate_trajectory(car_states)
    print(f"Reached episode return of {episode_return}.")

    # # Re-init Ray and load trained algorithm
    # ray.init(ignore_reinit_error=True)

    # restored_algo = Algorithm.from_checkpoint(checkpoint_dir)
    # env = CarWorldEnv()
    # obs, _ = env.reset()
    # done = False
    # total_reward = 0
    # dt = 0.05
    # t = 0
    # tf = 50
    # while (not done) and (t<tf):
    #     action = restored_algo.compute_single_action(obs)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    #     total_reward += reward
    #     done = terminated or truncated
    #     t += dt
    #     env.render()

    # print(f"Total reward from trained policy: {total_reward}")
    # env.close()
    # ray.shutdown()
