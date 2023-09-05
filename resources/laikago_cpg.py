import pybullet as p
import pybullet_data
import numpy as np
import math
from resources.cpg import CPG

class LaikagoCPG:
    '''
    VERSION WITH BAD REWARD, INSTANT FALL
    '''
    def __init__(self, client, dt=1./500):
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
        self.CPG = CPG(dt)
        self.pos = None
        self.ori = None
        self.vel = None
        self.last_pos = 0.052
        self.milestones = [i for i in range(1, 11)]
        self.max_episode_length = 5000

    def apply_action(self, action):
        # update CPG parameters
        self.CPG.update(action)
        # get CPG motor positions
        motor_angles = self.CPG.get_angles()
        for motor_id, i in zip(self.jointIds, range(len(motor_angles))):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=motor_angles[i],
                force=30,
            )

    def get_observation(self):
        self.pos, ori = p.getBasePositionAndOrientation(self.laikago)
        self.ori = p.getEulerFromQuaternion(ori)
        self.vel = p.getBaseVelocity(self.laikago)
        self.motor_positions = []
        for id in self.jointIds:
            self.motor_positions.append(p.getJointState(self.laikago, id)[0])
        cpg_params = [self.CPG._phases[0],
                      self.CPG._f,
                      self.CPG._Ah,
                      self.CPG._Ak_st,
                      self.CPG._Ak_sw,
                      self.CPG._d]
        return np.array(list(self.ori[0:2])+list(self.vel[1][0:2])+self.motor_positions+cpg_params)
    
    def calculate_reward(self, done, timestep):
        '''
        # milestone reward
        milestone_reward = 0
        # calculate step and global progress
        '''
        current_pos = self.pos[1]
        '''
        g_progress = current_pos - 0.052 #starting point in Y axis
        s_progress = current_pos - self.last_pos
        self.last_pos = current_pos
        # angular velocities
        a_vel = self.vel[1]
        
        # linear velocity
        l_vel = self.vel[0]
        '''
        # if early termination (proportional to how early it was)
        goal = 3.0
        if done:
            if current_pos > goal:
                return 0
            else: 
                return -1000 + timestep/5
        '''
        falling = (abs(a_vel[0])>2.0)
        if len(self.milestones)>0:
            if current_pos>self.milestones[0]:
                milestone_reward = self.milestones.pop(0)
        return 10*(l_vel[1]+milestone_reward)*(1-falling) - (abs(a_vel[1])+abs(a_vel[0]))
        '''
        return (current_pos-goal)
    
    def is_done(self, timestep):
        # robot falls
        if abs(self.ori[1]) > 0.5:
            return True
        if self.pos[2] < 0.3 or self.pos[1]<-0.2:
            return True
        # robot reaches goal
        if self.pos[1]>3.0:
            return True
        # maximum episode length
        if timestep >= self.max_episode_length:
            return True
        return False