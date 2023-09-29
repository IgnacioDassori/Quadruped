import numpy as np
import math

class CPG:
    def __init__(self, dt, gamma=5.0):
        # cpg static parameters
        self._dt = dt
        self._gamma = gamma
        # cpg adaptive parameters
        self._phases = np.array([0, math.pi, math.pi/2, 3*math.pi/2])
        self._f = 5.0
        self._Ah = 0.3
        self._Ak_st = 0.02
        self._Ak_sw = 0.5
        self._d = 0.5
        self._phase_lag = 0
        # offsets
        self._h_offset = 0.2 # 0.2 for 2 leg CPG version
        self._k_offset = 0.5 # 0.5 for 2 leg CPG version

    def update(self, updates):
        self._phases = (self._phases + 2*math.pi*self._f*self._dt) % (2*math.pi)
        # change parameters gradualy
        self._f += (updates[0]-self._f)*self._gamma*self._dt 
        self._Ah += (updates[1]-self._Ah)*self._gamma*self._dt
        self._Ak_st += (updates[2]-self._Ak_st)*self._gamma*self._dt
        self._Ak_sw += (updates[3]-self._Ak_sw)*self._gamma*self._dt
        self._d += (updates[4]-self._d)*self._gamma*self._dt
        
        # print current parameters
        #print(self._f, self._Ah, self._Ak_st, self._Ak_sw, self._d)
        '''
        self._phase_lag += (updates[5]-self._phase_lag)*self._gamma*self._dt
        # update phases
        self._phases[0:2] = (self._phases[0:2] + self._f*self._dt) % (2*math.pi)
        self._phases[2:4] = (self._phases[0:2] + self._f*self._dt + self._phase_lag) % (2*math.pi)
        '''

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
            if i<2:
                hip_angle += self._h_offset
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
            knee_angle = Ak*g + self._k_offset
            motor_angles.append(knee_angle)
        return np.array(motor_angles)*(-1)