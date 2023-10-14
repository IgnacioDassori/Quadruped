from stable_baselines3 import PPO
import gym
import os
from environments.plainCPG import plainCPGEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    # create log directory
    log_dir = "tmp/transfer_test/"
    os.makedirs(log_dir, exist_ok=True)

    # pretrained model
    version = "plainCPG2"
    gym_env = 'plainCPGEnv-v0'
    vec_env = DummyVecEnv([lambda: gym.make(gym_env)])
    vec_env = VecNormalize.load(f"tmp/{version}/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model_pretrained = PPO.load(f"tmp/{version}/best_model.zip")
    pretrained_policy = model_pretrained.policy
    print(pretrained_policy.state_dict())
    vec_env.close()

    # create quadruped environment
    env = DummyVecEnv([lambda: gym.make('plainCPGEnv-v0')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    for i in range(env.num_envs):
        env.envs[i] = Monitor(env.envs[i], log_dir)

    # create PPO model
    custom_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", n_steps=4096, batch_size=512,
                policy_kwargs=dict(net_arch=custom_arch))
    env.close()
    model.policy.load_state_dict(pretrained_policy.state_dict(), strict=False)
    for layer in model.policy.named_children():
        print(layer)
    