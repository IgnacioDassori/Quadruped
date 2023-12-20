import pybullet as p
import pybullet_data
import numpy as np
import math
import torch
from PIL import Image
from torchvision import transforms

class Laikago:
    '''
    AGENT FOR SLOPE ENVIRONMENT WITH ONLY PPO
    '''
    def __init__(self, client, start=0.0, dt=1./500):
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
        self.latent = None
        self.pos = None
        self.ori = None
        self.vel = None
        self.start = start
        self.last_pos = start
        self.max_episode_length = 10000
        self.goal = 10.0

    def spawn(self, pitch=0, z=0.44):
        quat = p.getQuaternionFromEuler([math.pi/2-pitch,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,z],
                                  quat, 
                                  flags = p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=False,
                                  physicsClientId=self.client)   

    def apply_action(self, action):
        # move motor by angles given by action
        for motor_id, i in zip(self.jointIds, range(len(action))):
            p.setJointMotorControl2(
                self.laikago,
                motor_id,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=30,
            )

    def get_observation(self, timestep, encoder):
        if timestep % 50 == 0:
            self.latent = self.get_latent_vector(encoder).squeeze(0).detach()
        self.pos, ori = p.getBasePositionAndOrientation(self.laikago)
        self.ori = p.getEulerFromQuaternion(ori)
        self.vel = p.getBaseVelocity(self.laikago)
        self.motor_positions = []
        for id in self.jointIds:
            self.motor_positions.append(p.getJointState(self.laikago, id)[0])
        observation = np.array(list(self.ori[0:2])+list(self.vel[1][0:2])+self.motor_positions)
        return np.concatenate((observation, self.latent))
    
    def get_latent_vector(self, encoder):
        # get latent vector from VAE, return as numpy array
        image = self.get_image() 
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
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
        camera_dist = 0.2
        camera_targ = 10000
        rotated = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), unit_vec)
        rotated_point = camera_dist*rotated + agent_pos
        rotated_view = camera_targ*rotated + agent_pos
        rotated_up = np.matmul(np.matmul(np.matmul(yaw_rot, pitch_rot), roll_rot), camera_up)
        # camera view matrix
        view_matrix = p.computeViewMatrix(rotated_point, rotated_view, rotated_up)
        # camera projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.1, farVal=30.0)
        # get camera image
        image = p.getCameraImage(width=128, height=128, viewMatrix=view_matrix, projectionMatrix=projection_matrix)[2]    
        return Image.fromarray(image).convert('RGB')
    
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
        
        falling = (abs(a_vel[0])>2.5)

        return 10*(l_vel[1])*(1-falling) - (abs(a_vel[1])+abs(a_vel[2]))
    
    def is_done(self, timestep):
        # robot falls
        if abs(self.ori[1]) > 0.5 or abs(self.ori[0]-math.pi/2) > 0.5:
            return True
        if (self.pos[1]-self.start)<-0.5:
            return True
        # robot reaches goal
        if self.pos[1]>self.goal:
            return True
        
        # test condition: Robot moves to the side too much
        if abs(self.pos[0]) > 1.0:
            return True

        # maximum episode length
        if timestep >= self.max_episode_length:
            return True
        return False