from stable_baselines3 import PPO
from environments.SlopeQuadrupedv1 import SlopeQuadrupedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    
    # create quadruped environment 
    env = SlopeQuadrupedEnv()

    # create PPO model
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    # train model
    model.learn(total_timesteps=1000000)

    # save model
    model.save("ppo_slopes")