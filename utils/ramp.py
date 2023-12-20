import pybullet as p
import os
import random
import math

class Ramp:
    def __init__(self, client):
        self._L = 2
        self._fname = os.path.join(os.path.dirname(__file__), f'urdf/ramp_{self._L}.urdf')
        self._client = client
        self.path_leght = 10
        # Lenght of segments
        self._ycord = self._L-1
        self._height = 0.0
        self._lowest = self._height
        self._pitch = 0.0
        self._currentType = 0 #0: no slope, 1: with slope

        #Parameters
        self._maxPitch = 10*math.pi/180
        self._minPitch = 5*math.pi/180
        self.generate()

    def generate(self):
        # load starting base
        p.loadURDF(self._fname, [0, self._L/2-1, 0])
        # load ramps
        for _ in range(self.path_leght):
            self.init_ramp()
        # load ending base
        p.loadURDF(self._fname, [0, self._ycord+self._L/2, self._height])

    def init_ramp(self):
        self.random_pitch()
        self._L = random.randint(2, 3)
        self._fname = os.path.join(os.path.dirname(__file__), f'urdf/ramp_{self._L}.urdf')
        bPosition = self.getBasePosition()
        bOrientation = p.getQuaternionFromEuler([self._pitch, 0, 0])
        p.loadURDF(fileName=self._fname,
                   basePosition=bPosition,
                   baseOrientation=bOrientation,
                   physicsClientId = self._client)
        
    def random_pitch(self):
        if self._currentType:
            self._pitch = 0.0
            self._currentType = 0
        else:
            sign = 1
            # sign = random.choice([1, -1])
            self._pitch = sign*random.uniform(self._minPitch, self._maxPitch)
            self._currentType = 1

    def getBasePosition(self):
        ycord = self._L*math.cos(self._pitch)/2
        height = self._L*math.sin(self._pitch)/2
        bPosition = [0, self._ycord+ycord, self._height+height]
        self._ycord += 2*ycord
        self._height += 2*height
        if self._height < self._lowest:
            self._lowest = self._height
        return bPosition
    
class Bridge:
    def __init__(self, client, pitch=None):
        self._fname = os.path.join(os.path.dirname(__file__), 'urdf/bridge.urdf')
        self._client = client
        if pitch is None:
            self.pitch = random.uniform(0, 5*math.pi/180) # uniform between 0 and 10 degrees
        else:
            self.pitch = pitch
        self.x_start = 4 - 5*math.sqrt(1-math.tan(self.pitch)**2)
        pos = [0, 4, 5*math.tan(self.pitch)]
        ori = p.getQuaternionFromEuler([self.pitch, 0, 0])
        p.loadURDF(fileName=self._fname, basePosition=pos, baseOrientation=ori, physicsClientId = self._client)
    
    def get_status(self):
        return self.pitch, -self.x_start*math.tan(self.pitch) + 0.44/math.cos(self.pitch)