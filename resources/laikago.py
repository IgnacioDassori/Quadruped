import pybullet as p
import pybullet_data
import numpy as np
import math
import torch


class Laikago:
    def __init__(self, client):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,.5],
                                  quat, 
                                  flags = p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=False,
                                  physicsClientId=self.client)
        # shoulder joints: 0, 4, 8, 12
        # upper leg joints: 1, 5, 9, 13
        # lower leg joints: 2, 6, 10, 14
        self.jointIds=[]
        i=0
        for _ in range(4):
            #self.jointIds.append(i)
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        self.latent = None
        self.orientation = None
        self.image = None
        self.motor_positions = None
        
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

    def get_observation(self, timestep, encoder):
        # get observation from high level every 50 steps (10Hz)
        if timestep % 50 == 0:
            # high level observation includes latent vector and orientation
            self.latent = self.get_latent_vector(encoder).squeeze(0)
            self.orientation = torch.tensor(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.laikago)[1]))
        # get observation from low level every step (500Hz)
        self.motor_positions = []
        for id in self.jointIds:
            self.motor_positions.append(p.getJointState(self.laikago, id)[0])
        l_velocity = list(p.getBaseVelocity(self.laikago)[0])
        a_velocity = list(p.getBaseVelocity(self.laikago)[1])
        # concatenate all observations
        return torch.cat((self.latent, self.orientation, torch.tensor(self.motor_positions+l_velocity+a_velocity)))

    def get_latent_vector(self, encoder):
        # get latent vector from VAE, return as numpy array
        image = self.get_image() # probably requieres different shape
        self.image = image
        image = (image - image.min()) / (image.max() - image.min())
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = encoder.encode(image)
        return encoder.reparameterize(mu, logvar)

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
        projection_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.1, farVal=100.0)
        # get camera image
        depth_img = p.getCameraImage(width=64, height=64, viewMatrix=view_matrix, projectionMatrix=projection_matrix)[3]    
        return depth_img

    def calculate_reward(self, done, timestep):
        # decrease penalty based on episode length
        if done:
            return -100
        vel = p.getBaseVelocity(self.laikago)[0]
        return vel[1]*timestep
    
    def is_done(self, timestep):
        # robot falls
        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.laikago)[1])
        if abs(orientation[0]-math.pi/2) > 0.5 or abs(orientation[1]) > 0.5:
            return True
        if timestep >= 5000:
            return True
        return False