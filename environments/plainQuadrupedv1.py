import gym 
import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from resources.laikago_tg import Laikago

class plainEnv(gym.Env):
    '''
    TG VERSION ENVIRONMENT
    '''
    def __init__(self):
        super(plainEnv, self).__init__()
        # TG parameters (ftg, atg, htg), motor position residual (x8)
        self.action_space = gym.spaces.box.Box(
            low=np.array([0.2, 0.0, -1.8, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),
            high=np.array([1.5, 1.5, -0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )
        # motor positions, roll & pitch (and rate), phase of TG
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-0.5, -2.0, -0.5, -2.0, -0.5, -2.0, -0.5, -2.0,
                          -math.pi, -math.pi, -10.0, -10.0, 0.0]),
            high=np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
                           math.pi, math.pi, 10.0, 10.0, 2*math.pi])
        )
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Lenght of timestep
        self._dt = 1./500
        p.setTimeStep(self._dt)
        self.timestep = 0
        p.setRealTimeSimulation(0)
        # Quadruped
        self.quadruped = None

    def step(self, action):
        # give action to agent
        if self.timestep == 0:
            time.sleep(1)
        self.quadruped.apply_action(action)
        p.stepSimulation()
        obs = self.quadruped.get_observation()
        done = self.quadruped.is_done(self.timestep)
        rew = self.quadruped.calculate_reward(done, self.timestep)
        info = {}
        self.timestep += 1
        return obs, rew, done, info

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.8)
        p.loadURDF("plane.urdf", basePosition=[0,0,0])
        self.quadruped = Laikago(client=self.client, dt=self._dt)
        # Define initial motor positions (radians)
        initial_motor_positions = [0.0, -0.8, 0.0, -0.8, 0.0, -0.8, 0.0, -0.8]
        # Set the initial motor positions
        for i, position in zip(self.quadruped.jointIds, initial_motor_positions):
            p.resetJointState(self.quadruped.laikago, jointIndex=i, targetValue=position)        
        self.timestep = 0
        return self.quadruped.get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)    