import os
import csv
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from typing import Callable

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 10 episodes
                mean_reward = np.mean(y[-10:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True
    
class SaveBestModelCallback(CheckpointCallback):
    def __init__(self, check_freq, save_path, name_prefix, vec_env, verbose=1):
        super().__init__(save_freq=check_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.best_mean_reward = -np.inf  # Initialize with negative infinity
        self.save_path = save_path
        self.check_freq = check_freq
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.check_freq == 0:
            rewards = [episode['r'] for episode in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            print("Num timesteps: {}".format(self.num_timesteps))
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save the model
                file_path = os.path.join(self.save_path, f"{self.name_prefix}.zip")
                print("Saving new best model to {}".format(file_path))
                self.model.save(file_path)
                # Save the VecNormalize statistics
                stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
                print(f"Saving VecNormalize to {stats_path}")
                self.vec_env.save(stats_path)
        return True
    
class MonitorCallback(BaseCallback):
    def __init__(self, check_freq, num_agents, save_path, name_prefix, vec_env, verbose=1):
        super().__init__(verbose=verbose)
        self.best_mean_reward = -np.inf  # Initialize with negative infinity
        self.name_prefix = name_prefix
        self.save_path = save_path
        self.check_freq = check_freq
        self.num_agents = num_agents
        self.vec_env = vec_env
        self.num_noimprove = 0

    def _on_step(self) -> bool:

        if self.num_noimprove >= 20:
            print("No improvement for 20 check_freq, stopping training")
            return False
        return True

    def _on_rollout_end(self) -> bool:
        R = 0
        E = 0
        for i in range(self.vec_env.num_envs):
            with open(f"{self.save_path}/monitor_{i}.csv", newline='') as f:
                rewards = 0
                timesteps = 0
                episodes = 0
                updates = 0
                reader = csv.DictReader(f)
                for row in reader:
                    rw = float(row['r'])
                    ts = float(row['l'])
                    if timesteps + ts > (self.check_freq/self.num_agents)*(updates+1):
                        rewards = 0
                        episodes = 0
                        updates += 1
                    rewards += rw
                    episodes += 1
                    timesteps += ts
            R += rewards
            E += episodes
        mean_reward = R/E
        print("Num timesteps: {}".format(self.num_timesteps))
        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            # Save the model
            file_path = os.path.join(self.save_path, f"{self.name_prefix}.zip")
            print("Saving new best model to {}".format(file_path))
            self.model.save(file_path)
            # Save the VecNormalize statistics
            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            print(f"Saving VecNormalize to {stats_path}")
            self.vec_env.save(stats_path)
            self.num_noimprove = 0
        else:
            self.num_noimprove += 1
            print(f"Updates with no improvement: {self.num_noimprove}")
        return True
    
def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
     
    def func(progress_remaining: float) -> float:

         
        return final_value + progress_remaining * (initial_value - final_value)
     
    return func