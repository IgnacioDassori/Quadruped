import gym 
import numpy as np
import torch
import math
import os
import json
import pybullet as p
import pybullet_data
from agents.laikago_ppo_slope import Laikago
from utils.ramp import Ramp
from VAE.modules import VAE

class slopeEnv(gym.Env):
    '''
    SLOPE ENVIRONMENT WITH CPG, NO MODULATION
    '''
    def __init__(self, mode, freq_range, gamma, vae_path):
        super(slopeEnv, self).__init__()
        # Motor positions (x8)
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.7, -1.0, -1.7, -1.0, -1.7, -1.0, -1.7]),
            high=np.array([1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3])
        )
        # roll, pitch, angular velocity (x4), motor positions (x8)
        # latent vector (x16)
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -10.0, -10.0,
                          -1.0, -1.7, -1.0, -1.7, -1.0, -1.7, -1.0, -1.7,
                          -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
                          -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0
                          ]),
            high=np.array([math.pi, math.pi, 10.0, 10.0,
                           1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 ,10.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 ,10.0
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
        self.quadruped = Laikago(client=self.client)
        self.quadruped.spawn()
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