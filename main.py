from stable_baselines3 import PPO
import gym
import pybullet as p
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from environments.modulating_slope import modulatingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":


    #version = "modulating_slopeEnv/even_bigger_updates_max4freq_biggernet_tr2"
    version = "modulating_plainEnv/walk_straight"
    # load config from json
    with open(f"results/{version}/config.json") as f:
        config = json.load(f)

    # plot the results
    df = pd.read_csv(f"results/{version}/monitor.csv", skiprows=1)
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
    use_vae = config['use_vae']
    if use_vae:
        vec_env = DummyVecEnv([lambda: gym.make(gym_env, mode=1, freq_range=config['freq_range'], gamma=config['gamma'], vae_path=config['vae_path'])])
    else:
        vec_env = DummyVecEnv([lambda: gym.make(gym_env, mode=1, freq_range=config['freq_range'], gamma=config['gamma'])])
    vec_env = VecNormalize.load(f"results/{version}/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load(f"results/{version}/best_model.zip")

    # evaluate model
    obs = vec_env.reset()
    cpg_params = []
    motor_positions = []
    R = []
    poses = []
    vel = []
    for i in range(30000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        time.sleep(0.002)
        pos = vec_env.envs[0].quadruped.pos
        poses.append(pos)
        R.append(rewards[0])
        quadruped = vec_env.envs[0].quadruped
        cpg = quadruped.CPG
        mp = []
        for id in quadruped.jointIds:
            mp.append(p.getJointState(quadruped.laikago, id)[0])
        motor_positions.append(mp)
        cpg_params.append([cpg._f, cpg._Ah, cpg._Ak_st, cpg._Ak_sw, cpg._d, cpg._off_h, cpg._off_k])
        if dones:
            rew = sum([0.99**i * r for i, r in enumerate(R)])
            for i in range(len(poses)-1):
                vel.append((poses[i+1][1]-poses[i][1])/0.002)
            
            

    
    plt.plot(cpg_params, label=['f', 'Ah', 'Ak_st', 'Ak_sw', 'd', 'off_h', 'off_k'])
    plt.xlabel('Timestep')
    plt.ylabel('Parameter Value')
    plt.title('CPG Parameters over Time')
    plt.legend()
    plt.show()
    
        
    plt.plot(motor_positions, label=['fl_hip', 'fl_knee', 'fr_hip', 'fr_knee', 'bl_hip', 'bl_knee', 'br_hip', 'br_knee'])
    plt.legend()
    plt.show()
    