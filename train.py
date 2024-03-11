import os
import gym
import json
import torch.nn as nn
from stable_baselines3 import PPO
from environments.modulating_slope import modulatingEnv
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from utils.callbacks import MonitorCallback, linear_schedule
        

if __name__ == "__main__":

    # create log directory

    environment = 'houseObstaclesEnv'
    experiment_name = 'test5'
    num_envs = 6
    use_vae = True
    if 'slope' in environment:
        use_vae = True
    log_dir = f"results_house/{environment}/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True) 

    # create quadruped environment
    freq_range = [1.5, 5.0]
    gamma = 10.0
    vae = 'dropout11'
    vae_path = os.path.join("VAE/results_house" if 'house' in environment else "VAE/results", vae)
    if use_vae:
        env = SubprocVecEnv([lambda i=i: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma, vae_path=vae_path, env_id=i, log_dir=log_dir) for i in range(num_envs)])
    else:
        env = lambda: gym.make(environment, mode=0, freq_range=freq_range, gamma=gamma)
    #env = DummyVecEnv([env_creator for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    '''
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)
    '''    
      
    # create PPO model
    n_steps = 8192*4
    batch_size = 1024*4
    lr = 0.0003
    tot_timesteps = 30000000
    activation = nn.ReLU
    custom_arch = dict(pi=[2048, 2048], vf=[2048, 2048])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", learning_rate=lr ,n_steps=n_steps, batch_size=batch_size,
                policy_kwargs=dict(activation_fn=activation,net_arch=custom_arch))

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
    #callback = SaveBestModelCallback(check_freq=n_steps, save_path=log_dir, name_prefix="best_model", vec_env=env ,verbose=1)
    callback = MonitorCallback(check_freq=n_steps*num_envs, num_agents=num_envs, save_path=log_dir, name_prefix="best_model", vec_env=env ,verbose=1)

    # train model
    model.learn(total_timesteps=tot_timesteps, callback=callback)
    env.close()