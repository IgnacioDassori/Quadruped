import os
import gym
from stable_baselines3 import PPO
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from resources.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":

    # create log directory
    log_dir = "tmp/plainCPG2/"
    os.makedirs(log_dir, exist_ok=True)

    # create quadruped environment
    env = DummyVecEnv([lambda: gym.make('plainCPGEnv-v0')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", n_steps=4096, batch_size=512)

    # create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=4096, log_dir=log_dir)

    # train model
    model.learn(total_timesteps=1000000, callback=callback)

    # save model
    stats_path = os.path.join(log_dir, "vec_normalize_cpg.pkl")
    env.save(stats_path)
    model.save("models/plainCPG2")
    env.close()