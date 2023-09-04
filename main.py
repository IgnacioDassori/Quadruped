from stable_baselines3 import PPO
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    # plot the results
    df = pd.read_csv("tmp/plainCPG2/monitor.csv", skiprows=1)
    rolling_mean_rewards = df['r'].rolling(window=10).mean()
    episode_numbers = np.arange(1, len(rolling_mean_rewards) + 1)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(df['r'], label='reward')
    axes[1].plot(episode_numbers, rolling_mean_rewards, label='rolling mean reward')
    plt.show()

    # load trained vector environment
    vec_env = DummyVecEnv([lambda: gym.make('plainCPGEnv-v0')])
    vec_env = VecNormalize.load("tmp/plainCPG2/vec_normalize_cpg.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load("tmp/plainCPG2/best_model.zip")

    # evaluate model
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)