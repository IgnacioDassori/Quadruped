import gym 
import numpy as np
import math
import os
import json
import torch
import csv
import pybullet as p
import pybullet_data
from VAE.modules import VAE_512
from utils.spawn_objects import SpawnManager
from agents.laikago_house_ppo import LaikagoHouse as Laikago

class houseEnv(gym.Env):
    '''
    HOUSE ENVIRONMENT PPO WITHOUT OBSTACLES, ADDED CORONAL MOTORS
    '''
    def __init__(self, mode, freq_range, gamma, vae_path, spawn_objects=False, env_id=None, log_dir=None):
        super(houseEnv, self).__init__()
        # Motor positions (x12)
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, 0, 0, 
                          0, 0, 0, 
                          0, 0, 0, 
                          0, 0, 0]),
            high=np.array([1, 1, 1, 
                           1, 1, 1, 
                           1, 1, 1, 
                           1, 1, 1])
        )
        # roll, pitch, yaw (x3), angular velocities (x3), motor positions (x12),
        # latent vector (X32)
        self.observation_space = gym.spaces.box.Box(
            low=np.array([0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0 ,0,
                          0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0 ,0
                          ]),
            high=np.array([1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1 ,1,
                           1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1 ,1                
                           ])
        )
        # start in GUI or DIRECT mode
        if mode:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Lenght of timestep
        p.setTimeStep(1./500)
        self.timestep = 0
        # Quadruped
        self.quadruped = None
        self.ramp = None
        # VAE model
        config_path = os.path.join(vae_path, "config.json")
        config = json.load(open(config_path))
        self.encoder = VAE_512(
            in_channels=config["in_channels"],
            latent_dim=config["latent_dim"],
            hidden_dims=config["layers"],
            output_activation=config["output_activation"]
        )
        self.encoder.load_state_dict(torch.load(os.path.join(vae_path, "best_model.pt")))
        self.encoder.eval()
        # Spawn objects
        self.spawn_objects = spawn_objects
        # Reward and episode length
        self.train = True
        if self.train:
            self.env_id = env_id
            self.log_dir = log_dir
            with open(f"{log_dir}/monitor_{env_id}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["r", "l"])
            self.reward = 0
            self.episode_length = 0

    def step(self, action):
        # give action to agent
        self.quadruped.apply_action(action)
        p.stepSimulation()
        obs = self.quadruped.get_observation(self.timestep, self.encoder)
        done = self.quadruped.is_done(self.timestep)
        rew = self.quadruped.calculate_reward(done, self.timestep)
        info = {}
        self.timestep += 1
        if self.train:
            self.reward += rew
            self.episode_length += 1
        return obs, rew, done, info

    def reset(self):
        # log reward and length
        if self.timestep > 0 and self.train:
            with open(f"{self.log_dir}/monitor_{self.env_id}.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.reward, self.episode_length])
            self.reward = 0
            self.episode_length = 0
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.8)
        self.quadruped = Laikago(client=self.client)
        sm = SpawnManager(spawn_objects=self.spawn_objects)
        start_pos = [sm.pos[0], sm.pos[1], 0.395]
        yaw = sm.angle
        self.quadruped.spawn(goal=sm.goal, yaw=yaw, start_pos=start_pos)
        # Initial motor positions
        initial_motor_positions = [0, 0.1, -0.9, 0, 0.1, -0.9, 0, 0.1, -0.9, 0, 0.1, -0.9]
        # Set the initial motor positions
        for i, position in zip(self.quadruped.jointIds, initial_motor_positions):
            p.resetJointState(self.quadruped.laikago, jointIndex=i, targetValue=position)     
        self.timestep = 0
        p.setRealTimeSimulation(0)
        return self.quadruped.get_observation(self.timestep, self.encoder)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)    