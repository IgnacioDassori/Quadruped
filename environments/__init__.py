from gym.envs.registration import register

register(
    id='plainCPGEnv-v0',
    entry_point='environments.plainCPG:plainCPGEnv',
    kwargs={'mode': 0, 'freq_range': [3.0, 8.0], 'gamma': 5.0}
)

register(
    id='bridgeEnv-v0',
    entry_point='environments.bridgeCPG:oneSlopeEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='plainCPGEnv-v1',
    entry_point='environments.plainCPG_v1:plainCPGEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='fullEnv-v0',
    entry_point='environments.fullEnvironment:fullEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'lr5e-3_bs16_kld0.00025'}    
)

register(
    id='plainCPGEnv-v2',
    entry_point='environments.plainCPG_v2:plainCPGEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='modulatingEnv-v0',
    entry_point='environments.modulating:modulatingEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='modulatingEnv-v1',
    entry_point='environments.modulating_v1:modulatingEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'lr5e-3_bs16_kld0.00025'}
)