import gym 
import numpy as np
import time
import math
import torch
import os
import json
import pybullet as p
import pybullet_data
from resources.laikago_modulating_v2 import LaikagoCPG as Laikago
from resources.ramp import Ramp
from VAE.modules import VAE

class modulatingEnv(gym.Env):
    '''
    MODULATING WITH SLOPE ENVIRONMENT USING VAE AND DIFFERENT OFFSETS FOR BACK AND FRONT LEGS
    '''
    def __init__(self, mode=0, freq_range=[3.0, 8.0], gamma=5.0, vae_path=None):
        super(modulatingEnv, self).__init__()
        # CPG parameters (f, Ah, Ak_st, Ak_sw, d, off_h_b, off_k_b, off_h_f, off_k_f) (x9)
        # Motor position correction (x8)
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, 0.1, 0.0, 0.1, 0.5, 0.0, 0.3, 0.0, 0.3,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 0.5, 1.0, 0.95, 0.4, 0.7, 0.4, 0.7,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )
        # roll, pitch, angular velocity (x4), motor positions (x8), phase & CPG parameters (x10)
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -10.0, -10.0,
                          -1.0, -1.7, -1.0, -1.7, -1.0, -1.7, -1.0, -1.7,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.3
                          ]),
            high=np.array([math.pi, math.pi, 10.0, 10.0,
                           1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3,
                           2*math.pi, 10.0, 1.0, 0.5, 1.0, 0.95, 0.4, 0.7, 0.4, 0.7
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
        # CPG proportional controler gain
        self.gamma = gamma
        self.freq_range = freq_range
        # VAE model
        config_path = os.path.join(vae_path, "config.json")
        config = json.load(open(config_path))
        self.encoder = VAE(
            in_channels=config["in_channels"],
            latent_dim=config["latent_dim"]
        )
        self.encoder.load_state_dict(torch.load(os.path.join(vae_path, "best_model.pt")))
        self.encoder.eval()

    def step(self, action):
        # give action to agent
        self.quadruped.apply_action(action)
        p.stepSimulation()
        obs = self.quadruped.get_observation(self.timestep, self.encoder)
        done = self.quadruped.is_done(self.timestep)
        rew = self.quadruped.calculate_reward(done, self.timestep)
        info = {}
        self.timestep += 1
        return obs, rew, done, info

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.8)
        self.ramp = Ramp(client=self.client)
        p.loadURDF("plane.urdf", basePosition=[0,0,self.ramp._lowest])
        self.quadruped = Laikago(client=self.client, gamma=self.gamma)
        self.quadruped.spawn()
        self.quadruped.CPG.freq_range = self.freq_range
        # Define initial motor positions (radians)
        initial_motor_positions = [0, -0.7, 0, -0.7, 0, -0.7, 0, -0.7]
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