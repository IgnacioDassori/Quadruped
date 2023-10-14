import os
import gym
import json
from stable_baselines3 import PPO
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from resources.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":

    # create log directory
    log_dir = "tmp/plainCPG_recreate3/"
    os.makedirs(log_dir, exist_ok=True)

    # create quadruped environment
    environment = 'plainCPGEnv-v0'
    env = DummyVecEnv([lambda: gym.make(environment)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    n_steps = 4096
    batch_size = 512
    custom_arch = dict(pi=[64, 64], vf=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", 
                n_steps=n_steps, batch_size=batch_size, 
                policy_kwargs=dict(net_arch=custom_arch))

    # create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=4096, log_dir=log_dir)

    # train model
    model.learn(total_timesteps=1000000, callback=callback)

    # save model
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    env.close()

    # save config
    config = dict(
        environment=environment,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=custom_arch
    )
    with open(os.path.join(log_dir, "config.json"), 'w') as fp:
        json.dump(config, fp, indent=4)