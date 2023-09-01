import pybullet as p
import pybullet_data
import numpy as np
import math
from resources.trajectory_generator import TrajectoryGenerator


class Laikago:
    def __init__(self, client, dt):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,.43],
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
        self.TG = TrajectoryGenerator(dt)
        
    def apply_action(self, action):
        # update TG parameters
        self.TG.update(action[0:3])
        # get TG motor positions
        motor_angles = self.TG.get_angles()
        for motor_id, i in zip(self.jointIds, range(len(motor_angles))):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=motor_angles[i]+action[i+3],
                force=20
            )

    def get_observation(self):
        # get observation
        self.motor_positions = []
        for id in self.jointIds:
            self.motor_positions.append(p.getJointState(self.laikago, id)[0])
        orientation = list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.laikago)[1])[0:2])
        a_velocity = list((p.getBaseVelocity(self.laikago)[1])[0:2])
        # concatenate all observations
        return np.array(self.motor_positions + orientation + a_velocity + [self.TG._TG_phase])

    def calculate_reward(self, done, timestep):
        # penalize early terminations, reward episode length
        if done:
            return -100 + timestep/50
        # reward forward movement, penalize angular velocity
        vel = p.getBaseVelocity(self.laikago)[0]
        return vel[1]*20 - abs(self.orientation[0]-math.pi/2)*10 - abs(self.orientation[1])*10
    
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