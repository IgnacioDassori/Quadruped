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
    log_dir = "tmp/full/first_test"
    os.makedirs(log_dir, exist_ok=True) 

    # create quadruped environment
    freq_range = [1.5, 5]
    gamma = 10.0
    environment = 'fullEnv-v0'
    vae = 'lr5e-3_bs16_kld0.00025'
    vae_path = os.path.join("VAE/tmp_eval", vae)
    env = DummyVecEnv([lambda: gym.make(environment, mode=1, freq_range=freq_range, gamma=gamma, vae_path=vae_path)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    n_steps = 4096*4
    batch_size = 512*2
    lr = 0.0003
    tot_timesteps = 5000000
    activation = nn.Tanh
    custom_arch = dict(pi=[128, 128], vf=[128, 128])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=lr ,n_steps=n_steps, batch_size=batch_size,
                policy_kwargs=dict(activation_fn=activation,net_arch=custom_arch), tensorboard_log=log_dir)

    # save config json
    config = dict(
        env=environment,
        vae=vae,
        lr=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        net_arch=custom_arch,
        activation_fn=str(activation),
        total_timesteps=tot_timesteps,
        freq_range=freq_range,
        gamma=gamma
    )
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        f.write(json_object)

    # create callback
    #callback = SaveOnBestTrainingRewardCallback(check_freq=n_steps, log_dir=log_dir)
    callback = SaveBestModelCallback(check_freq=n_steps, save_path=log_dir, name_prefix="best_model", verbose=1)

    # train model
    model.learn(total_timesteps=tot_timesteps, callback=callback)

    # save model
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    env.close()