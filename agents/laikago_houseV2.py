import pybullet as p
import pybullet_data
import numpy as np
import math
import torch
from PIL import Image
from torchvision import transforms
from utils.cpg import CPG_house as CPG

class LaikagoHouse:
    '''
    AGENT FOR HOUSE V2 ENVIRONMENT
    '''
    def __init__(self, client, dt=1./500, gamma=5.0):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.laikago = None
        # coronal leg joints: 0, 4, 8, 12
        # upper leg joints: 1, 5, 9, 13
        # lower leg joints: 2, 6, 10, 14
        self.jointIds=[]
        i=0
        for _ in range(4):
            self.jointIds.append(i)
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        self.CPG = CPG(dt, gamma)
        self.pos = None
        self.ori = None
        self.vel = None
        self.start_pos = None
        self.goal = None
        self.goal_vector = None
        self.angle_to_goal = 0
        self.max_episode_length = 8000
        self.lower_obs = np.array([-math.pi/2, -math.pi/2, 0, -10.0, -10.0, -10.0,
                                   -0.1, -0.4, -1.4, -0.1, -0.4, -1.4, -0.1, -0.4, -1.4, -0.1, -0.4, -1.4,
                                   0, 0, 0, 0, 0.5,
                                   -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250,
                                   -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250, -250
        ])
        self.upper_obs = np.array([math.pi/2, math.pi/2, math.pi, 10.0, 10.0, 10.0,
                                   0.1, 0.6, -0.4, 0.1, 0.6, -0.4, 0.1, 0.6, -0.4, 0.1, 0.6, -0.4,
                                   2*math.pi, 0.5, 0.2, 0.5, 0.95,
                                   250, 250, 250, 250, 250, 250, 250 ,250, 250, 250, 250, 250, 250, 250, 250, 250,
                                   250, 250, 250, 250, 250, 250, 250 ,250, 250, 250, 250, 250, 250, 250, 250, 250
        ])
        self.lower_act = np.array([0.0, 0.0, 0.0, 0.5,
                                   -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                                   -0.1, -0.1, -0.1, -0.1
        ])
        self.upper_act = np.array([0.5, 0.2, 0.5, 0.95,
                                   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                   0.1, 0.1, 0.1, 0.1
        ])            
        self.last_pos = None

    def spawn(self, goal, yaw=0, start_pos=[0,0,0.44]):
        self.goal = goal
        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi+yaw])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  start_pos,
                                  quat, 
                                  flags = p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=False,
                                  physicsClientId=self.client)
        
    def apply_action(self, action):
        # rescale the actions
        action = np.array(action)*(self.upper_act - self.lower_act) + self.lower_act
        cpg_params = action[:4]
        motor_corrections = action[4:12]
        coronal_angles = action[12:]
        # update CPG parameters
        self.CPG.update(np.concatenate((cpg_params, coronal_angles)))
        # get CPG motor positions
        motor_angles = self.CPG.get_angles()
        # move motors
        for i, motor_id in enumerate(self.jointIds):
            counter = 0
            if motor_id%4 == 0:
                p.setJointMotorControl2(
                    self.laikago,
                    motor_id,
                    p.POSITION_CONTROL,
                    targetPosition=motor_angles[i],
                    force=30,
                )
            else:
                p.setJointMotorControl2(
                    self.laikago,
                    motor_id,
                    p.POSITION_CONTROL,
                    targetPosition=motor_angles[i] + motor_corrections[counter],
                    force=30,
                )
                counter += 1
        

    def get_observation(self, timestep, encoder):
        if timestep % 50 == 0:
            self.latent = self.get_latent_vector(encoder).squeeze(0).detach()
        self.pos, ori = p.getBasePositionAndOrientation(self.laikago)
        if self.start_pos is None:
            self.start_pos = self.pos[:2]
            self.goal_vector = [
                self.euclidean_dist(self.start_pos, self.goal),
                self.angle_diff(self.start_pos, self.goal)
            ]
        self.ori = list(p.getEulerFromQuaternion(ori))
        self.ori[0] -= math.pi/2
        self.ori[2] = self.ori[2] + math.pi*(3/2) if self.ori[2] < -math.pi/2 else self.ori[2] - math.pi/2
        self.vel = p.getBaseVelocity(self.laikago)
        self.motor_positions = []
        for id in self.jointIds:
            motor_pos = p.getJointState(self.laikago, id)[0]
            if id%8 == 0:
                motor_pos *= -1
            self.motor_positions.append(motor_pos)
        cpg_params = [self.CPG._phases[0],
                      self.CPG._Ah,
                      self.CPG._Ak_st,
                      self.CPG._Ak_sw,
                      self.CPG._d
                      ]
        observation = np.array(self.ori+list(self.vel[1])+self.motor_positions+cpg_params)
        observation = np.concatenate((observation, self.latent))
        # normalize observations to [0, 1]
        norm_obs = (observation - self.lower_obs)/(self.upper_obs - self.lower_obs)
        return norm_obs
    
    def get_latent_vector(self, encoder):
        # get latent vector from VAE, return as numpy array
        image = self.get_image() 
        image = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = encoder.encode(image)
        return mu
    
    def get_image(self):
        # position and orientation of the agent
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.laikago)
        euler = p.getEulerFromQuaternion(agent_orn)
        roll, pitch, yaw = euler
        # rotation matrices
        roll_rot = np.array(([1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]))
        pitch_rot = np.array(([math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]))
        yaw_rot = np.array(([math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]))
        unit_vec = np.array([0, 0, 1])
        camera_up = np.array([0, 1, 0])
        camera_dist = 1
        camera_targ = 10000
        rotated = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), unit_vec)
        rotated_point = camera_dist*rotated + agent_pos
        rotated_view = camera_targ*rotated + agent_pos
        rotated_up = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), camera_up)
        # camera view matrix
        view_matrix = p.computeViewMatrix(rotated_point, rotated_view, rotated_up)
        # camera projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.1, farVal=30.0)
        # get camera image
        image = p.getCameraImage(width=512, height=512, viewMatrix=view_matrix, projectionMatrix=projection_matrix)[2]    
        return Image.fromarray(image).convert('RGB')
    
    def calculate_reward(self, done, timestep):

        # w is the ratio between progress and goal distance
        X = self.euclidean_dist(self.start_pos, self.pos[:2])
        G = self.goal_vector[0]
        w = X/G

        # reward term 1: forward progress
        l_vel = self.vel[0]
        a_vel = self.vel[1]
        r1 = min(l_vel[1], 1.0)

        # reward term 2: velocity in the goal direction
        # theta: robot orientation (rotated to 0 in x-axis)
        # phi: angle of vector from robot to goal
        theta = self.ori[2]
        phi = self.angle_diff(self.pos[:2], self.goal)
        vel_norm = self.euclidean_dist([0,0], l_vel)
        vel_angle = self.angle_diff([0,0], l_vel)

        r2 = min(vel_norm, 1.0) * math.cos(vel_angle - phi)
        if abs(vel_angle - theta) > 0.5:
            r2 = min(r2, 0)

        # when episode ends: penalty or reward
        # penalty proportional to how far from goal and length of episode
        r3 = 0
        if done:
            if self.euclidean_dist(self.pos[:2], self.goal) < 2.0:
                r3 = 1000
            else:
                #r3 = -100 + w*50 + 50*timestep/self.max_episode_length
                r3 = -1000 + w*500 + 500*timestep/self.max_episode_length

        # penalize instability
        r4 = abs(a_vel[0]) + abs(a_vel[1])

        # penalize looking away from goal
        self.angle_to_goal = abs(theta%(2*math.pi)-phi)
        r5 = abs((theta%(2*math.pi)-phi))**2

        # nullify rewards when falling
        #falling = (abs(a_vel[0])>2.0 or abs(a_vel[1])>2.0)   

        # term weight depends on w
        #R = (1-w)*r1 + 2*w*r2 + r3 - 0.1*r4 - 5*w*r5
        R = max(1-2*w,0)*r1 + min(2*w,1)*r2 + r3 - 0.3*r4 - 3*r5
        #print(f"R: {R}, r1: {r1}, r2: {r2}, r3: {r3}, r4: {r4}, r5: {r5}")

        return R


    def euclidean_dist(self, X, Y):

        return math.sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)
    
    def angle_diff(self, X, Y):

        return math.atan2(Y[1]-X[1], Y[0]-X[0])
    
    def is_done(self, timestep):

        # robot falls
        if abs(self.ori[1]) > 0.5 or abs(self.ori[0]) > 0.5:
            return True
        # robot reaches goal
        if self.euclidean_dist(self.pos[:2], self.goal) < 2.0:
            return True
        # robot approaches a wall
        if self.pos[0] > 4.0 or self.pos[0] < -4.0 or self.pos[1] > 4.0:
            return True
        # robot is not facing the goal by a certain angle
        if self.angle_to_goal > math.pi/4:
            return True
        # robot falls backwards
        if self.pos[1]<-4.6:
            return True
        # maximum episode length
        if timestep >= self.max_episode_length:
            return True
        return False