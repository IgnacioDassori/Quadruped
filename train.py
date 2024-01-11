import os
import gym
import json
import torch.nn as nn
from stable_baselines3 import PPO
from environments.modulating_slope import modulatingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.callbacks import SaveBestModelCallback
from stable_baselines3.common.monitor import Monitor
        

if __name__ == "__main__":

    # create log directory

    environment = 'modulating_plainEnv'
    experiment_name = 'new_reward_max3freq'
    use_vae = False
    if 'slope' in environment:
        use_vae = True
    log_dir = f"results_new_rew/{environment}/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True) 

    # create quadruped environment
    freq_range = [1.0, 3.0]
    gamma = 10.0
    vae = 'lr5e-3_bs16_kld0.00025'
    vae_path = os.path.join("VAE/results", vae)
    if use_vae:
        env = DummyVecEnv([lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma, vae_path=vae_path)])
    else:
        env = DummyVecEnv([lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    n_steps = 4096*16
    batch_size = 512*16
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
    callback = SaveBestModelCallback(check_freq=n_steps, save_path=log_dir, name_prefix="best_model", vec_env=env ,verbose=1)

    # train model
    model.learn(total_timesteps=tot_timesteps, callback=callback)
    env.close()