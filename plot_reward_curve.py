import matplotlib.pyplot as plt
import numpy as np
import csv
import math

if __name__ == '__main__':

    dir = "results_house/houseObstaclesEnv/test5"
    tot_timesteps = 100000
    n_steps = 8192*4
    agents = 6


    R = []
    E = []
    for i in range(agents):
        with open(f"{dir}/monitor_{i}.csv", newline='') as f:
            rewards = 0
            timesteps = 0
            episodes = 0
            updates = 0
            reader = csv.DictReader(f)
            for row in reader:
                rw = float(row['r'])
                ts = float(row['l'])
                if timesteps + ts > (n_steps)*(updates+1):
                    if i == 0:
                        R.append(rewards)
                        E.append(episodes)
                    else:
                        R[updates] += rewards
                        E[updates] += episodes
                    rewards = 0
                    episodes = 0
                    updates += 1
                rewards += rw
                episodes += 1
                timesteps += ts
        if i == 0:
            R.append(rewards)
            E.append(episodes)
        else:
            R[updates] += rewards
            E[updates] += episodes
    R = np.array(R)
    E = np.array(E)
    mean_reward = R/E
    print(mean_reward.max())
    print(len(mean_reward)*n_steps*agents)
    plt.plot(mean_reward)
    plt.show()
'''

    T = np.array([n_steps*agents*i for i in range(1, 1+math.ceil(tot_timesteps/(n_steps*agents)))])
    R = np.zeros_like(T)
    N = np.zeros_like(T)

    for i in range(6):
        timesteps = 0
        rewards = 0
        n = 0
        episodes = 0
        with open(f"{dir}/monitor_{i}.csv", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rw=float(row['r'])
                ts=float(row['l'])
                if timesteps + ts > n_steps*(n+1):
                    R[n] += rewards
                    N[n] += episodes
                    n += 1
                    rewards = 0
                rewards += rw
                timesteps += ts
                episodes += 1
            R[n] += rewards
            N[n] += episodes

plt.plot(T, R/N)
plt.show()
'''