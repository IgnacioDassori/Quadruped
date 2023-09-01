import gym 
import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from resources.laikago_plain import Laikago

class plainEnv(gym.Env):

    def __init__(self):
        super(plainEnv, self).__init__()
        # motor position (x8)
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.5, -2.0, -0.5, -2.0]),
            high=np.array([0.5, -0.6, 0.5, -0.6])
        )
        # pitch, motor positions (x8), velocity, angular velocity
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -math.pi,
                          -0.5, -2.0, -0.5, -2.0,
                          0.0, 0.0, 0.0, 0.0,
                          -10.0, -10.0]),
            high=np.array([math.pi, math.pi,
                           0.5, -0.6, 0.5, -0.6,
                           3.0, 3.0, 3.0, 3.0,
                           10.0, 10.0])
        )
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Lenght of timestep
        p.setTimeStep(1./500)
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
        self.quadruped = Laikago(client=self.client)
        # Define initial motor positions (radians)
        initial_motor_positions = [0.0, -0.7, 0.0, -0.7, 0.0, -0.7, 0.0, -0.7]
        # Set the initial motor positions
        for i, position in zip(self.quadruped.jointIds, initial_motor_positions):
            p.resetJointState(self.quadruped.laikago, jointIndex=i, targetValue=position)
            p.resetJointState(self.quadruped.laikago, jointIndex=i+4, targetValue=position)      
        self.timestep = 0
        return self.quadruped.get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect(self.client)    