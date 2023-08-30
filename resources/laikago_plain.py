import pybullet as p
import pybullet_data
import numpy as np
import math


class Laikago:
    def __init__(self, client):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,.45],
                                  quat, 
                                  flags = p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=False,
                                  physicsClientId=self.client)
        # upper leg joints: 1, 5, 9, 13
        # lower leg joints: 2, 6, 10, 14
        self.jointIds=[]
        i=0
        for _ in range(4):
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        self.motor_positions = None
        self.orientation = None
        
    def apply_action(self, action):
        # actions are the motor positions
        for motor_id, motor_position in zip(self.jointIds, action):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=motor_position,
                force=20
            )

    def get_observation(self):
        # get observation
        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.laikago)[1])
        self.motor_positions = []
        for id in self.jointIds:
            self.motor_positions.append(p.getJointState(self.laikago, id)[0])
        l_velocity = list(p.getBaseVelocity(self.laikago)[0])
        a_velocity = list(p.getBaseVelocity(self.laikago)[1])
        # concatenate all observations
        return np.array(list(orientation)+self.motor_positions+l_velocity+a_velocity)

    def calculate_reward(self, done, timestep):
        # penalize early terminations, reward episode length
        if done:
            return -100 + timestep/50
        # reward forward movement, penalize angular velocity
        vel = p.getBaseVelocity(self.laikago)[0]
        return vel[1]*100 - abs(vel[0])*10 - abs(self.orientation[0]-math.pi/2)*10 - abs(self.orientation[1])*10
    
    def is_done(self, timestep):
        # robot falls
        pos, ori = p.getBasePositionAndOrientation(self.laikago)
        self.orientation = p.getEulerFromQuaternion(ori)
        if abs(self.orientation[0]-math.pi/2) > 0.5 or abs(self.orientation[1]) > 0.5:
            return True
        if pos[2] < 0.2:
            return True
        # maximum episode length
        if timestep >= 5000:
            return True
        return False