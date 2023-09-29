from stable_baselines3 import PPO
import gym
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    version = "tensor_test"
    # load config from json
    with open(f"tmp/{version}/config.json") as f:
        config = json.load(f)

    # plot the results
    df = pd.read_csv(f"tmp/{version}/monitor.csv", skiprows=1)
    rolling_mean_rewards = df['r'].rolling(window=10).mean()
    episode_numbers = np.arange(1, len(rolling_mean_rewards) + 1)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(df['r'], label='reward')
    axes[0].set_title('Reward per episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[1].plot(episode_numbers, rolling_mean_rewards, label='rolling mean reward')
    axes[1].set_title('Rolling mean reward (last 10) per episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Rolling mean reward')
    plt.show()

    # load trained vector environment
    gym_env = config['env']
    vec_env = DummyVecEnv([lambda: gym.make(gym_env, mode=1, freq_range=config['freq_range'], gamma=config['gamma'])])
    vec_env = VecNormalize.load(f"tmp/{version}/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load(f"tmp/{version}/best_model.zip")

    # evaluate model
    obs = vec_env.reset()
    cpg_params = []
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        cpg = vec_env.envs[0].quadruped.CPG
        cpg_params.append([cpg._f, cpg._Ah, cpg._Ak_st, cpg._Ak_sw, cpg._d])
        if dones:
            break
        
    plt.plot(cpg_params, label=['f', 'Ah', 'Ak_st', 'Ak_sw', 'd'])
    plt.legend()
    plt.show()
        