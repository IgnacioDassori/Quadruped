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

    log_dir = "tmp/modulating_v2/new_net_10e7ts"
    os.makedirs(log_dir, exist_ok=True) 

    # create quadruped environment
    freq_range = [1.5, 5]
    gamma = 10.0
    environment = 'modulatingEnv-v2'
    use_vae = True
    vae = 'lr5e-3_bs16_kld0.00025'
    vae_path = os.path.join("VAE/tmp_eval", vae)
    if use_vae:
        env = DummyVecEnv([lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma, vae_path=vae_path)])
    else:
        env = DummyVecEnv([lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    n_steps = 4096*4
    batch_size = 512*4
    lr = 0.0002
    tot_timesteps = 10000000
    activation = nn.Tanh
    custom_arch = dict(pi=[256, 128], vf=[256, 128])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=lr ,n_steps=n_steps, batch_size=batch_size,
                policy_kwargs=dict(activation_fn=activation,net_arch=custom_arch), tensorboard_log=log_dir)

    # save config json
    config = dict(
        env=environment,
        vae_path=vae_path,
        lr=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        net_arch=custom_arch,
        activation_fn=str(activation),
        total_timesteps=tot_timesteps,
        freq_range=freq_range,
        gamma=gamma,
        use_vae=use_vae
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