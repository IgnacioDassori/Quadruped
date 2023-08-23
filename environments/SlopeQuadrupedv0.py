import gym 
import numpy as np
import math
import pybullet as p
from resources.ramp import Ramp
from resources.laikago import Laikago

class SlopeQuadrupedEnv(gym.Env):

    def __init__(self):
        super(SlopeQuadrupedEnv, self).__init__()
        # motor positions (x12)
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.5, -0.5, -2.0, -0.5, -0.5, -2.0, -0.5, -0.5, -2.0, -0.5, -0.5, -2.0]),
            high=np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0])
        )
        # latent vector (x8), motor positions (x12), velocity, angular velocity, pitch
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-0.5, -0.5, -2.0, -0.5, -0.5, -2.0, -0.5, -0.5, -2.0, -0.5, -0.5, -2.0]),
            high=np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0])
        )
        self.np_random, _ = gym.utils.seeding.np_random()
        
        self.client = p.connect(p.GUI)
        # Lenght of timestep
        p.setTimeStep(1./500)
        # World and quadruped
        self.ramp = None
        self.quadruped = None
        self.reset()

    def step(self, action):
        # give action to agent
        self.quadruped.apply_action(action)
        # get HL observation every 50 steps (10Hz)
        obs = self.quadruped.get_observation()
        rew = self.quadruped.calculate_reward()
        done = self.quadruped.is_done()
        return obs, rew, done, info

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.8)
        self.ramp = Ramp(client=self.client)
        self.laikago = Laikago(client=self.client)

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]