import math
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, dt):
        # starting phase of FL
        self._TG_phase = 0
        self._phase_diff = math.pi/2
        # adaptive parameters
        self._ftg = 0.5
        self._alfa = 1.0
        self._htg = -0.8
        # TG parameters
        self._dt = dt
        self._Cs = 0.0
        self._Ae = 0.2
        self._theta = 0.2
        self._beta = 0.7

    def update(self, updates):
        self._TG_phase = (self._TG_phase + 2*math.pi*self._ftg*self._dt) % (2*math.pi)
        self._ftg = updates[0]
        self._alfa = updates[1]
        self._htg = updates[2]

    def get_angles(self):
        # check if leg is in stance or swing phase
        phases = [(self._TG_phase + i*self._phase_diff) % (2*math.pi) for i in range(4)]
        motor_angles = []
        for phi in phases:
            # swing phase
            if phi < 2*math.pi*self._beta:
                t = phi/(2*self._beta)
            # stance phase
            else:
                t = (phi + 2*math.pi*(1-2*self._beta))/(2*(1-self._beta))
            # calculate angles
            motor_angles.append(self._Cs + self._alfa*math.cos(t))
            motor_angles.append(self._htg + self._Ae*math.sin(t) + self._theta*math.cos(t))
        return motor_angles
    
if __name__ == "__main__":
    TG = TrajectoryGenerator(0.002)
    s_fl = []
    e_fl = []
    for i in range(4000):
        TG.update([0.5, 1.0, -0.8])
        motor_angles = TG.get_angles()
        s_fl.append(motor_angles[0])
        e_fl.append(motor_angles[1])

fig, axes = plt.subplots(2, 1)

axes[0].plot(s_fl)
axes[1].plot(e_fl)
plt.show()