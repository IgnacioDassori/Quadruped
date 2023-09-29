import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

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
    def __init__(self, save_path, name_prefix, verbose=1):
        super().__init__(save_freq=4096, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.best_mean_reward = -np.inf  # Initialize with negative infinity
        self.save_path = os.path.join(save_path, "best_model")

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            rewards = [episode['r'] for episode in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            print("Num timesteps: {}".format(self.num_timesteps))
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print("Saving new best model to {}".format(self.save_path))
                self.model.save(self.save_path)
        return True

