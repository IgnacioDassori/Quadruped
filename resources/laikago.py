import pybullet as p
import math

class Laikago:
    def __init__(self, client):
        self.client = client
        urdfFlags = p.URDF_USE_SELF_COLLISION
        quat = p.getQuaternionFromEuler([math.pi/2,0,math.pi])
        self.laikago = p.loadURDF("laikago/laikago_toes.urdf",
                                  [0,0,.5],
                                  quat, 
                                  flags = urdfFlags,
                                  useFixedBase=False)
        # shoulder joints: 0, 4, 8, 12
        # upper leg joints: 1, 5, 9, 13
        # lower leg joints: 2, 6, 10, 14
        self.jointIds=[]
        i=0
        for n in range(4):
            self.jointIds.append(i)
            self.jointIds.append(i+1)
            self.jointIds.append(i+2)
            i+=4
        self.latent_vector = None
        
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
        # get observation from high level every 50 steps (10Hz)
        # get observation from low level every step (500Hz)
        pass

    def get_image(self):
        pass

    def calculate_reward(self):
        pass
    
    def is_done(self):
        pass