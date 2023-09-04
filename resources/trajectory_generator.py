import math
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, dt):
        # starting phase of FL
        self._TG_phase = 0
        self._phase_diff = 0
        # adaptive parameters
        self._ftg = 1.0
        self._alfa = 1.0
        self._htg = 0.0
        # TG parameters
        self._dt = dt
        self._Cs = 0.0
        self._Ae = 1.0
        self._theta = 1.0
        self._beta = 0.75

    def update(self, updates):
        self._TG_phase = (self._TG_phase + 2*math.pi*self._ftg*self._dt) % (2*math.pi)
        self._ftg = updates[0]
        self._alfa = updates[1]
        self._htg = updates[2]

    def get_angles(self):
        # check if leg is in stance or swing phase
        phases = [self._TG_phase]
        for phi in phases:
            # swing phase
            if phi < 2*math.pi*self._beta:
                t = phi/(2*self._beta)
            # stance phase
            else:
                t = (phi + 2*math.pi*(1-2*self._beta))/(2*(1-self._beta))
            # calculate angles
            s = self._Cs + self._alfa*math.cos(t)
            e = self._htg + self._Ae*math.sin(t) + self._theta*math.cos(t)
        return s, e, phi
    
if __name__ == "__main__":
    TG = TrajectoryGenerator(0.002)
    S = []
    E = []
    Phi = []
    for i in range(499):
        TG.update([1.0, 1.0, 0.0])
        s, e, phi = TG.get_angles()
        S.append(s)
        E.append(e)
        Phi.append(phi)

print(TG._TG_phase)
fig, axes = plt.subplots(2, 1)

axes[0].plot(Phi,S)
axes[1].plot(Phi,E)
plt.show()