import os
import gym
from stable_baselines3 import PPO
from environments.plainQuadrupedv0 import plainEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from resources.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # create quadruped environment
    #env = plainEnv()
    env = DummyVecEnv([lambda: gym.make('plainQuadruped-v0')])
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
    log_dir = "tmp/"
    stats_path = os.path.join(log_dir, "vec_normalize_plain.pkl")
    model.save("models/plain")
    env.save(stats_path)
    env.close()