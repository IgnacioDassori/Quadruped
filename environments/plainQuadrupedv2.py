import gym 
import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from resources.laikago_cpg import LaikagoCPG as Laikago

class plainEnv(gym.Env):
    '''
    CPG VERSION ENVIRONMENT
    '''
    def __init__(self):
        super(plainEnv, self).__init__()
        # CPG parameters (w, Ah, Ak_st, Ak_sw, d, phase_lag)
        self.action_space = gym.spaces.box.Box(
            low=np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([20.0, 1.0, 0.8, 1.0, 1.0, math.pi*2])
        )
        # roll, pitch, angular velocity (x2), motor positions (x8), CPG parameters & phase (x5)
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi, -10.0, -10.0,
                          -1.0, -1.8, -1.0, -1.8, -1.0, -1.8, -1.0, -1.8,
                          0.0, 1.0, 0.0, 0.0, 0.0, 0.0
                          ]),
            high=np.array([math.pi, math.pi, 10.0, 10.0,
                           1.0, 0.2, 1.0, 0.2, 1.0, 0.2, 1.0, 0.2,
                           2*math.pi, 3.0, 1.0, 0.8, 1.0, 1.0
                           ])
        )
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Lenght of timestep
        p.setTimeStep(1./500)
        self.timestep = 0
        # Quadruped
        self.quadruped = None

    def step(self, action):
        if self.timestep == 0:
            self.quadruped.CPG._f = action[0]
            self.quadruped.CPG._Ah = action[1]
            self.quadruped.CPG._Ak_st = action[2]
            self.quadruped.CPG._Ak_sw = action[3]
            self.quadruped.CPG._d = action[4]
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
        self.quadruped = Laikago(client=self.client)
        # Define initial motor positions (radians)
        initial_motor_positions = [-0.6, -0.8, -0.6, -0.8, -0.4, -0.8, -0.4, -0.8]
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