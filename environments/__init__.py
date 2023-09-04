from gym.envs.registration import register

register(
    id='plainQuadruped-v0',
    entry_point='environments.plainQuadrupedv0:plainEnv'
)

register(
    id='plainQuadruped-v1',
    entry_point='environments.plainQuadrupedv1:plainEnv'
)

register(
    id='plainQuadruped-v2',
    entry_point='environments.plainQuadrupedv2:plainEnv'
)

register(
    id='plainCPGEnv-v0',
    entry_point='environments.plainCPG:plainCPGEnv'
)