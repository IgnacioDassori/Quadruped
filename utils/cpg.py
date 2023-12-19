import numpy as np
import math

class CPG:
    def __init__(self, dt, gamma=5.0):
        # cpg static parameters
        self._dt = dt
        self._gamma = gamma
        # cpg adaptive parameters
        self._phases = np.array([0, math.pi, math.pi/2, 3*math.pi/2])
        self.freq_range = None
        '''
        # DETERMINISTIC STARTING PARAMETERS
        self._f = 5.0
        self._Ah = 0.3
        self._Ak_st = 0.02
        self._Ak_sw = 0.5
        self._d = 0.5

        '''
        # RANDOM STARTING PARAMETERS
        self._f = np.random.uniform(1.5, 5.0)
        self._Ah = np.random.uniform(0.1, 0.5)
        self._Ak_st = np.random.uniform(0.01, 0.2)
        self._Ak_sw = np.random.uniform(0.3, 0.7)
        self._d = np.random.uniform(0.3, 0.7)
        '''
        # original offsets
        self._off_h = 0.2 # 0.2 for 2 leg CPG version
        self._off_k = 0.5 # 0.5 for 2 leg CPG version
        '''
        # new offsets
        self._off_h = 0.0
        self._off_k = 0.7
        '''
        TRY LATER
        # offsets
        self._off_h = np.random.uniform(0.0, 0.4)
        self._off_k = np.random.uniform(0.3, 0.7)
        '''



    def update(self, updates):
        self._phases = (self._phases + 2*math.pi*self._f*self._dt) % (2*math.pi)
        # change parameters gradualy
        freq_action = updates[0]*(self.freq_range[1]-self.freq_range[0]) + self.freq_range[0]
        self._f += (freq_action-self._f)*self._gamma*self._dt 
        self._Ah += (updates[1]-self._Ah)*self._gamma*self._dt
        self._Ak_st += (updates[2]-self._Ak_st)*self._gamma*self._dt
        self._Ak_sw += (updates[3]-self._Ak_sw)*self._gamma*self._dt
        self._d += (updates[4]-self._d)*self._gamma*self._dt
        self._off_h += (updates[5]-self._off_h)*self._gamma*self._dt
        self._off_k += (updates[6]-self._off_k)*self._gamma*self._dt

    def get_angles(self):
        # maybe change to make all calculations on an array
        motor_angles = []
        for phi, i in zip(self._phases, range(len(self._phases))):
            # hip phase
            if phi < 2*math.pi*self._d:
                phi_h = phi/(2*self._d)
            else:
                phi_h = (phi + 2*math.pi*(1-2*self._d))/(2*(1-self._d))
            # hip angle
            hip_angle = self._Ah*math.cos(phi_h)
            '''
            if i<2:
                hip_angle += self._off_h
            '''
            hip_angle += self._off_h    
            motor_angles.append(hip_angle)
            # knee amplitude
            phi_k = (phi_h) % (2*math.pi)
            if phi_k < math.pi:
                Ak = self._Ak_st
            else:
                Ak = self._Ak_sw
            # cubic profile
            theta = 2*( ( phi_k / (2*math.pi) ) % 0.5)
            if theta < 0.5:
                g = -16*(theta**3) + 12*(theta**2)
            else:
                g = 16*(theta-0.5)**3 - 12*(theta-0.5)**2 + 1
            # knee angle
            knee_angle = Ak*g + self._off_k
            motor_angles.append(knee_angle)
        return np.array(motor_angles)*(-1)