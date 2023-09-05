import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pybullet as p


# GLOBAL PARAMETERS
f = 5.0
d = 0.75
off_h = 0
off_k = 0.8

# HIP JOINT
timesteps = 5000
end_time = 1
t = np.linspace(0, end_time, timesteps)
Phi = np.array([(2*math.pi*f*i*t[1])%(2*math.pi) for i in range(timesteps)])

Ah = 0.6
phi_h = []
for phi in Phi:
    phi = phi % (2*math.pi)
    if phi < 2*math.pi*d:
        phi_h.append(phi/(2*d))
    else:
        phi_h.append((phi + 2*math.pi*(1-2*d))/(2*(1-d)))
hip_commands = []
for phi in phi_h:
    hip_commands.append((Ah*math.cos(phi)+off_h)*(-1))

# KNEE JOINT
phi_k = phi_h # can add offset later
Ak_st = 0.5
Ak_sw = 0.8
ak = []
for phi in phi_k:
    if phi <  math.pi:
        ak.append(Ak_st)
    else:
        ak.append(Ak_sw)
gamma = []
T = []
for phi in phi_k:
    theta = 2*( ( phi / (2*math.pi) ) % 0.5)
    T.append(theta)
    if theta < 0.5:
        g = -16*(theta**3) + 12*(theta**2)
    else:
        g = 12*(theta-0.5)**3 - 12*(theta-0.5)**2 + 1
    gamma.append(max(g,0))
knee_commands = []
for i in range(len(gamma)):
    knee_commands.append((ak[i]*gamma[i]+off_k)*(-1))

fig, axes = plt.subplots(2, 1)
axes[0].plot(t, hip_commands)
axes[0].set_title("Hip Commands")
axes[0].xlim = (0, end_time)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Angle (rad)")
axes[1].plot(t, knee_commands)
axes[1].set_title("Knee Commands")
axes[1].xlim = (0, end_time)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angle (rad)")
plt.show()


# animation of trajectory
Ru = 0.5 # radius upper leg
Rl = 0.3 # radius lower leg
u_offset = 0
l_offset = 0

fig, ax = plt.subplots()
line = ax.plot([], [], 'b')[0]
line2 = ax.plot([], [], 'r')[0]
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

xu = []
yu = []
for angle in hip_commands:
    angle += u_offset
    xu.append(Ru*math.cos(angle))
    yu.append(Ru*math.sin(angle))

xl = []
yl = []
for angle in knee_commands:
    xl.append(Rl*math.cos(angle))
    yl.append(Rl*math.sin(angle))

def update(frame):
    line.set_data([0, xu[frame]], [0, yu[frame]])
    line2.set_data([xu[frame], xu[frame]-xl[frame]], [yu[frame], yu[frame]-yl[frame]])
    return line,line2

# Create the animation
ani = FuncAnimation(fig, update, frames=timesteps, blit=True, interval=2)
plt.show()