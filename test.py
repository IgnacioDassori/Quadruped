import os
import gym
import json
import torch.nn as nn
import pybullet as p
from stable_baselines3 import PPO
from environments.fullEnvironment import fullEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from resources.callbacks import SaveOnBestTrainingRewardCallback, SaveBestModelCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

if __name__ == "__main__":

    freq_range = [1.5, 5]
    gamma = 10.0
    vae = 'lr5e-3_bs16_kld0.00025'
    vae_path = os.path.join("VAE/tmp_eval", vae)

    env = fullEnv(mode=1, freq_range=freq_range, gamma=gamma, vae_path=vae_path)
    obs = env.reset()