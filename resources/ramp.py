import pybullet as p
import os
import random
import math

class Ramp:
    def __init__(self, client):
        self._fname = os.path.join(os.path.dirname(__file__), 'ramp.urdf')
        self._client = client
        self.path_leght = 3
        # Lenght of segments
        self.L = 4
        self._ycord = self.L-1
        self._height = 0.0
        self._pitch = 0.0
        self._currentType = 0 #0: no slope, 1: with slope

        #Parameters
        self._maxPitch = 25*math.pi/180
        self._minPitch = 15*math.pi/180
        self.generate()

    def generate(self):
        # load starting base
        p.loadURDF(self._fname, [0, self.L/2-1, 0])
        # load ramps
        for _ in range(self.path_leght):
            self.init_ramp()
        # load ending base
        p.loadURDF(self._fname, [0, self._ycord+self.L/2, self._height])

    def init_ramp(self):
        self.random_pitch()
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
            sign = random.choice([1, -1])
            self._pitch = sign*random.uniform(self._minPitch, self._maxPitch)
            self._currentType = 1

    def getBasePosition(self):
        ycord = self.L*math.cos(self._pitch)/2
        height = self.L*math.sin(self._pitch)/2
        bPosition = [0, self._ycord+ycord, self._height+height]
        self._ycord += 2*ycord
        self._height += 2*height
        return bPosition