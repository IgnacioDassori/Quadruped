from stable_baselines3 import PPO
from environments.SlopeQuadrupedv0 import SlopeQuadrupedEnv

if __name__ == "__main__":
    # create quadruped environment
    env = SlopeQuadrupedEnv()

    # create PPO model
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    # train model
    model.learn(total_timesteps=10000)

    # save model
    model.save("ppo_slopes")