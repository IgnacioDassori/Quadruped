import torch.nn as nn
from stable_baselines3 import PPO
import gym
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

log_dir = "tmp/test/"

env = DummyVecEnv([lambda: gym.make('plainCPGEnv-v0')])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
for i in range(env.num_envs):
    env.envs[i] = Monitor(env.envs[i], log_dir)

policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

for layer in model.policy.children():
    print(layer)