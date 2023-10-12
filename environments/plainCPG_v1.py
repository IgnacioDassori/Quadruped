import gym 
import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from resources.laikago_cpg_4legs_v1 import LaikagoCPG as Laikago

class plainCPGEnv(gym.Env):
    '''
    CPG 4 LEGS ENVIRONMENT. VERSION THAT ADDS OFFSET ACTION AND OBSERVATION
    '''
    def __init__(self, mode=0, freq_range=[3.0, 8.0], gamma=5.0):
        super(plainCPGEnv, self).__init__()
        # CPG parameters (f, Ah, Ak_st, Ak_sw, d, off_h, off_k)
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.3]),
            high=np.array([1.0, 1.0, 0.8, 1.0, 0.95, 0.4, 0.7])
        )
        # roll, pitch, angular velocity (x2), motor positions (x8), phase & CPG parameters
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -10.0, -10.0,
                          -1.0, -1.7, -1.0, -1.7, -1.0, -1.7, -1.0, -1.7,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3
                          ]),
            high=np.array([math.pi, math.pi, 10.0, 10.0,
                           1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3,
                           2*math.pi, 10.0, 1.0, 0.8, 1.0, 1.0, 0.4, 0.7
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

    def step(self, action):
        # give action to agent
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
        return self.quadruped.get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)    