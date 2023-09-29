import gym 
import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from resources.laikago_cpg_4legs import LaikagoCPG as Laikago
from resources.ramp import Ramp

class plainCPGEnv(gym.Env):
    '''
    CPG 4 LEGS ENVIRONMENT. FINAL VERSION
    '''
    def __init__(self, start=0.0):
        super(plainCPGEnv, self).__init__()
        # CPG parameters (f, Ah, Ak_st, Ak_sw, d)
        self.action_space = gym.spaces.box.Box(
            low=np.array([3.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([8.0, 1.0, 0.8, 1.0, 1.0])
        )
        # roll, pitch, angular velocity (x2), motor positions (x8), CPG parameters & phase
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -10.0, -10.0,
                          -1.0, -1.7, -1.0, -1.7, -1.0, -1.7, -1.0, -1.7,
                          0.0, 3.0, 0.0, 0.0, 0.0, 0.0
                          ]),
            high=np.array([math.pi, math.pi, 10.0, 10.0,
                           1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0, 0.3,
                           2*math.pi, 8.0, 1.0, 0.8, 1.0, 1.0
                           ])
        )
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Lenght of timestep
        p.setTimeStep(1./500)
        self.timestep = 0
        self.start = start
        # Quadruped
        self.quadruped = None

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
        ramp = Ramp(client=self.client)
        for _ in range(3):
            ramp.init_ramp()
        #p.loadURDF("plane.urdf", basePosition=[0,0,0])
        self.quadruped = Laikago(client=self.client, start=self.start)
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