import gym 
import numpy as np
import math
import torch
import pybullet as p
import sys
sys.path.append('..')
from resources.ramp import Ramp
from resources.laikago import Laikago
from VAE.modules import VAE

class SlopeQuadrupedEnv(gym.Env):

    def __init__(self):
        super(SlopeQuadrupedEnv, self).__init__()
        # motor positions (x8)
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.5, -2.0, -0.5, -2.0, -0.5, -2.0, -0.5, -2.0]),
            high=np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
        )
        # latent vector (x8), pitch, motor positions (x8), velocity, angular velocity
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                          -math.pi, -math.pi, -math.pi,
                          -0.5, -2.0, -0.5, -2.0, -0.5, -2.0, -0.5, -2.0,
                          -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           math.pi, math.pi, math.pi,
                           0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
                           10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        )
        self.client = p.connect(p.DIRECT)
        # Lenght of timestep
        p.setTimeStep(1./500)
        self.timestep = 0
        p.setRealTimeSimulation(0)
        # World and quadruped
        self.ramp = None
        self.quadruped = None
        # VAE model
        self.encoder = VAE()
        self.encoder.load_state_dict(torch.load("models/model.pt"))
        self.encoder.eval()
        # Check if a CUDA device is available
        '''
        if torch.cuda.is_available():
            self.encoder.to('cuda')
        '''

    def step(self, action):
        # give action to agent
        self.quadruped.apply_action(action)
        p.stepSimulation()
        # get HL observation every 50 steps (10Hz)
        obs = self.quadruped.get_observation(self.timestep, self.encoder)
        done = self.quadruped.is_done(self.timestep)
        rew = self.quadruped.calculate_reward(done, timestep=self.timestep)
        info = {}
        self.timestep += 1
        return obs, rew, done, info

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.8)
        self.ramp = Ramp(client=self.client)
        self.quadruped = Laikago(client=self.client)
        self.timestep = 0
        return self.quadruped.get_observation(self.timestep, self.encoder)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)