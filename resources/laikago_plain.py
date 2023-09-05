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
        '''
        for _ in range(4):
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        '''
        ''' FOR TWO LEG VERSION (V2)'''
        for _ in range(2):
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=8
        
        self.orientation = None
        self.current_pos = p.getBasePositionAndOrientation(self.laikago)[0][1]
        
    def apply_action(self, action):
        # actions are the motor positions
        for motor_id, motor_position in zip(self.jointIds, action):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=motor_position,
                force=20, 
                maxVelocity=3,
            )
            ''' FOR TWO LEG VERSION (V2) '''
            p.setJointMotorControl2(
                self.laikago,
                motor_id+4,
                p.POSITION_CONTROL,
                targetPosition=motor_position,
                force=20, 
                maxVelocity=3,
            )
            

    def get_observation(self):
        # get observation
        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.laikago)[1])[0:2]
        motor_positions = []
        motor_velocities = []
        for id in self.jointIds:
            pos, vel = p.getJointState(self.laikago, id)[0:2]
            motor_positions.append(pos)
            motor_velocities.append(vel)
        a_velocity = list(p.getBaseVelocity(self.laikago)[1][0:2])
        # concatenate all observations
        return np.array(list(orientation)+motor_positions+motor_velocities+a_velocity)

    def calculate_reward(self, done, timestep):
        # curriculum reward
        milestone = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
        milestone_reward = 0
        # penalize early terminations, reward episode length
        pos, _ = p.getBasePositionAndOrientation(self.laikago)
        g_progress = pos[1] - 0.052 #starting point in Y axis
        c_progress = pos[1] - self.current_pos
        # reward forward movement, penalize angular velocity
        _, a_vel = p.getBaseVelocity(self.laikago)
        if done:
            return -100 + g_progress*10 + timestep/50 - abs(a_vel[0])**2
        self.current_pos = pos[1]
        not_falling = (abs(a_vel[0])<2.0)
        if self.current_pos>milestone[0]:
            milestone_reward = milestone.pop(0)
        return (c_progress*200+milestone_reward*100)*not_falling - abs(a_vel[0])**2 - abs(a_vel[1]) - (1-not_falling)*500
        '''
        #sparse rewards
        pos, _ = p.getBasePositionAndOrientation(self.laikago)
        milestones = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
        milestone_reward = 0
        if done:
            return -100 + timestep/50
        if pos[1]>milestones[0]:
            milestone_reward = milestones.pop(0)
        _, a_vel = p.getBaseVelocity(self.laikago)
        return milestone_reward*100 - abs(a_vel[0]) - abs(a_vel[1])
        '''
        
    
    def is_done(self, timestep):
        # robot falls
        pos, ori = p.getBasePositionAndOrientation(self.laikago)
        self.orientation = p.getEulerFromQuaternion(ori)
        if abs(self.orientation[0]-math.pi/2) > 0.5 or abs(self.orientation[1]) > 0.5:
            return True
        if pos[2] < 0.3 or pos[1]<-0.1:
            return True
        # maximum episode length
        if timestep >= 5000:
            return True
        return False