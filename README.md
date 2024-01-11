# Quadruped Gait Adaptation via CPG and PPO

<div style="text-align: justify;">

This repository contains the code developed for the Memoria of Ignacio Dassori for the grade of Electrical Engineering, Universidad de Chile. The code is structured as follows:

## Environments

In the environments folder the code for all the different pybullet env's which were used for testing our proposed system. Environments are separated between what type of terrain the quadruped has to traverse (plain or slope), aswell as the strategy employed (ppo, cpg, modulating). The environment is a class of type gym.Env, necessary for its working with pybullet. When initializing the class, observation and action spaces must be defined.

Other methods required are the step, reset and close methods. The render method can also be included, although it is not necessary as pybullet offers a GUI. The step method applies actions by performing a simulation step, and returns observation, rewards and the state of the episode (done). How exactly actions are applied, observations obtained and rewards calculated are defined in the agent code, not in the environment class.

The reset method is responsible for reseting to the initial state after an episode is finished, initializing a new trial. This includes spawning the robot and any other object such as obstacles (3D objects and urdf's). It must return the observation at the first time step. In our case, we spawn the robot with some random elements so that there is variation in the starting conditions.

Finally, the close method just disconnects the pybullet client.

## Agents

## VAE

## Utils

### CPG

### Callbacks

### Ramp

### Objects

## Results

## Performance

## Other useful

</div>