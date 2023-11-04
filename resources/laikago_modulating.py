import pybullet as p
import pybullet_data
import numpy as np
import math
from resources.cpg import CPG

class LaikagoCPG:
    '''
    VERSION THAT INCLUDES MODULATION OF MOTOR POSITIONS. 
    GOES WITH MODULATINGCPG.PY ENVIRONMENT
    '''
    def __init__(self, client, start= 0.0, dt=1./500, gamma=5.0):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.laikago = None
        # upper leg joints: 1, 5, 9, 13
        # lower leg joints: 2, 6, 10, 14
        self.jointIds=[]
        i=0
        for _ in range(4):
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        self.CPG = CPG(dt, gamma)
        self.pos = None
        self.ori = None
        self.vel = None
        self.start = start
        self.last_pos = start
        self.max_episode_length = 5000
        self.goal = 5.0
        self.modulation_range = [-0.1, 0.1]

    def spawn(self, pitch=0, z=0.44):
        quat = p.getQuaternionFromEuler([math.pi/2-pitch,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,z],
                                  quat, 
                                  flags = p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=False,
                                  physicsClientId=self.client)   

    def apply_action(self, action):
        # update CPG parameters
        cpg_params = action[:7]
        motor_corrections = [a*(self.modulation_range[1]-self.modulation_range[0]) + self.modulation_range[0] for a in action[7:]]
        #motor_corrections = action[7:]
        self.CPG.update(cpg_params)
        # get CPG motor positions
        motor_angles = self.CPG.get_angles()
        for motor_id, i in zip(self.jointIds, range(len(motor_angles))):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=motor_angles[i] + motor_corrections[i],
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
                      self.CPG._d,
                      self.CPG._off_h,
                      self.CPG._off_k
                      ]
        return np.array(list(self.ori[0:2])+list(self.vel[1][0:2])+self.motor_positions+cpg_params)
    
    def calculate_reward(self, done, timestep):

        current_pos = self.pos[1]

        # angular velocities
        a_vel = self.vel[1]
        
        # linear velocity
        l_vel = self.vel[0]
        
        # if early termination (proportional to how early it was)
        if done:
            if current_pos > self.goal:
                return 0
            else: 
                return -1000 + timestep/(self.max_episode_length/1000)
        
        falling = (abs(a_vel[0])>2.0)

        return 10*(l_vel[1])*(1-falling) - (abs(a_vel[1])+abs(a_vel[2]))
    
    def calculate_reward_2(self):

        current_pos = self.pos[1]
        vcap = 1.5
        velocity= (current_pos - self.last_pos)/0.002

        return max(-vcap, min(velocity, vcap))
    
    def is_done(self, timestep):
        # robot falls
        if abs(self.ori[1]) > 0.5 or abs(self.ori[0]-math.pi/2) > 0.5:
            return True
        if (self.pos[1]-self.start)<-0.5:
            return True
        # robot reaches goal
        if self.pos[1]>self.goal:
            return True
        # maximum episode length
        if timestep >= self.max_episode_length:
            return True
        return False