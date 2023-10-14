from stable_baselines3 import PPO
import gym
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    version = "plainCPG_recreate3"

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
    gym_env = 'plainCPGEnv-v0'
    vec_env = DummyVecEnv([lambda: gym.make(gym_env)])
    vec_env = VecNormalize.load(f"tmp/{version}/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load(f"tmp/{version}/best_model.zip")
    model.policy.training = False

    # evaluate model
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        