from gym.envs.registration import register

register(
    id='plainQuadruped-v0',
    entry_point='environments.plainQuadrupedv0:plainEnv'
)