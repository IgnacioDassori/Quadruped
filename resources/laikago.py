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
        
    def apply_action(self, action):
        pass

    def get_observation(self):
        pass