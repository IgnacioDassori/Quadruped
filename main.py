from stable_baselines3 import PPO
from environments.plainQuadrupedv0 import plainEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == "__main__":

    # load trained vector environment
    vec_env = DummyVecEnv([lambda: plainEnv()])
    vec_env = VecNormalize.load("tmp/vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # load PPO model
    model = PPO.load("models/ppo_plain2")

    # evaluate model
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)