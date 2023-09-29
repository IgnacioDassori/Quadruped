import os
import gym
import json
import torch.nn as nn
from stable_baselines3 import PPO
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from resources.callbacks import SaveOnBestTrainingRewardCallback, SaveBestModelCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
        

if __name__ == "__main__":

    # create log directory
    log_dir = "tmp/tensor_test/"
    os.makedirs(log_dir, exist_ok=True)

    # create quadruped environment
    freq_range = [5, 10]
    gamma = 10.0
    environment = 'plainCPGEnv-v0'
    env = DummyVecEnv([lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    n_steps = 4096
    batch_size = 512
    lr = 0.0003
    tot_timesteps = 1000000
    custom_arch = dict(pi=[64, 64], vf=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=lr ,n_steps=n_steps, batch_size=batch_size,
                policy_kwargs=dict(net_arch=custom_arch), tensorboard_log=log_dir)

    # save config json
    config = dict(
        env=environment,
        lr=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=custom_arch,
        total_timesteps=tot_timesteps,
        freq_range=freq_range,
        gamma=gamma
    )
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # create callback
    #callback = SaveOnBestTrainingRewardCallback(check_freq=n_steps, log_dir=log_dir)
    callback = SaveBestModelCallback(save_path=log_dir, name_prefix="best_model", verbose=1)

    # train model
    model.learn(total_timesteps=tot_timesteps, callback=callback)

    # save model
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    env.close()