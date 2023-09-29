from gym.envs.registration import register

register(
    id='slopeQuadruped-v1',
    entry_point='environments.SlopeQuadrupedv1:SlopeQuadrupedEnv'
)

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
    entry_point='environments.plainCPG:plainCPGEnv',
    kwargs={'mode': 0, 'freq_range': [3.0, 8.0], 'gamma': 5.0}
)

register(
    id='slopeCPGEnv-v0',
    entry_point='environments.slopeCPG:slopeCPGEnv',
    kwargs={'start': 0.0}
)