import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_histograms(data_dir):
    '''
    Function that plots histograms of the three method's performance
    It creates an histogram of the last position and of the mean lenght
    of a successful run
    data_dir: directory containing .npy files with the trials information
    '''

    # Set plot style
    sns.set(style="whitegrid", context="paper", font_scale=1.5)
   

    # Load data
    data_list = os.listdir(f"{data_dir}")
    for file in data_list:
        if 'ppo' in file:
            if 'time' in file:
                ppo_time = np.load(f"{data_dir}/{file}")
            else:
                ppo_pos = np.load(f"{data_dir}/{file}")
        elif 'cpg' in file:
            if 'time' in file:
                cpg_time = np.load(f"{data_dir}/{file}")
            else:
                cpg_pos = np.load(f"{data_dir}/{file}")
        elif 'mod' in file:
            if 'time' in file:
                mod_time = np.load(f"{data_dir}/{file}")
            else:
                mod_pos = np.load(f"{data_dir}/{file}")

    # Plot 3 histograms of Final Position [m] reached
    custom_bins = list(range(0, 11))
    fig, axs = plt.subplots(1, 3, figsize=(12, 7))
    axs[0].hist(ppo_pos, bins=custom_bins, color='#1f77b4')
    axs[0].title.set_text('PPO')
    axs[0].set_xlabel('Final Position [m]')
    axs[0].set_ylabel('Frequency')
    axs[0].set_ylim([0,50])
    axs[1].hist(cpg_pos, bins=custom_bins, color='#1f77b4')
    axs[1].title.set_text('PPO + CPG')
    axs[1].set_xlabel('Final Position [m]')
    axs[1].set_ylim([0,50])
    axs[2].hist(mod_pos, bins=custom_bins, color='#1f77b4')
    axs[2].title.set_text('PPO + CPG + mod')
    axs[2].set_xlabel('Final Position [m]')
    axs[2].set_ylim([0,50])
    fig.suptitle('Final position for 50 trials in Slope Environment, smaller updates', fontsize=20)
    plt.show()

if __name__ == "__main__":
    
    data_dir = "performance/slope_smaller_update"
    plot_histograms(data_dir)