from gym.envs.registration import register

register(
    id='ppo_plainEnv',
    entry_point='environments.ppo_plain:plainEnv',
    kwargs={'mode': 0, 'freq_range': None, 'gamma': None}
)

register(
    id='ppo_slopeEnv',
    entry_point='environments.ppo_slope:slopeEnv',
    kwargs={'mode': 0, 'freq_range': None, 'gamma': None, 'vae_path': 'lr5e-3_bs16_kld0.00025'}
)

register(
    id='cpg_plainEnv',
    entry_point='environments.cpg_plain:plainCPGEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='cpg_slopeEnv',
    entry_point='environments.cpg_slope:slopeCPGEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'lr5e-3_bs16_kld0.00025'}    
)

register(
    id='modulating_plainEnv',
    entry_point='environments.modulating_plain:modulatingEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0}
)

register(
    id='modulating_slopeEnv',
    entry_point='environments.modulating_slope:modulatingEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'lr5e-3_bs16_kld0.00025'}
)

register(
    id='houseEnv',
    entry_point='environments.house:houseEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'dropout11', 'spawn_objects': False,
            'env_id': 0, 'log_dir': None}
)

register(
    id='houseEnvV2',
    entry_point='environments.houseV2:houseEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'dropout11', 'spawn_objects': False,
            'env_id': 0, 'log_dir': None}
)

register(
    id='houseObstaclesEnv',
    entry_point='environments.house_obstacles:houseEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'dropout11', 'spawn_objects': True,
            'env_id': 0, 'log_dir': None}
)

register(
    id='housePPOEnv',
    entry_point='environments.house_ppo:houseEnv',
    kwargs={'mode': 0, 'freq_range': None, 'gamma': None, 'vae_path': 'dropout11', 'spawn_objects': False,
            'env_id': 0, 'log_dir': None}
)

register(
    id='houseEnvV3',
    entry_point='environments.houseV3:houseEnv',
    kwargs={'mode': 0, 'freq_range': [1.5, 5.0], 'gamma': 10.0, 'vae_path': 'dropout11', 'spawn_objects': False,
            'env_id': 0, 'log_dir': None}
)