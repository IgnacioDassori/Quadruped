# I need to plot the reward curve inside a csv file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_label(filename):

    if 'ppo' in filename:
        return 'PPO'
    elif 'cpg' in filename:
        return 'PPO + CPG'
    else:
        return 'PPO + CPG + mod'

def plot_csv(csv_dir):

    colors = ["b", "r", "g", "y", "m", "k", "c"]

    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    # Create a line plot
    plt.figure(figsize=(10, 6))

    csv_list = os.listdir(f"csv/{csv_dir}")
    temp = csv_list[0]
    csv_list[0] = csv_list[1]
    csv_list[1] = temp

    for i, file in enumerate(csv_list):
        exec(f"df{i} = pd.read_csv('csv/{csv_dir}/{file}')")
        exec(f"steps{i} = df{i}['Step']")
        exec(f"rew{i} = df{i}['Value']")
        exec(f"sns.lineplot(x=steps{i}, y=rew{i}, color=colors[{i}], label='{generate_label(file)}', linewidth=2, alpha=0.9)")

    # Add labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve for Plain Environment')

    low = min(pd.read_csv(f"csv/{csv_dir}/{csv_list[0]}")['Step'])
    plt.xlim(low, 10**7)

    # Add legend
    plt.legend()

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or show the plot
    #plt.savefig('reward_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_np(np_dir):

    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    # Create a line plot
    plt.figure(figsize=(10, 6))

    train_data = np.load(f"{np_dir}/train_loss.npy")
    eval_data = np.load(f"{np_dir}/eval_loss.npy")

    sns.lineplot(train_data, color="b", label="Training Loss", linewidth=2, alpha=0.9)
    sns.lineplot(eval_data, color="r", label="Eval Loss", linewidth=2, alpha=0.9)

    plt.xlim(1, len(train_data)-1)
    plt.ylim(0, eval_data[2])

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch-wise Loss for VAE Training and Validation')

    # Add legend
    plt.legend()

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or show the plot
    #plt.savefig('reward_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    csv_dir = "plain_hyperparameters"
    np_dir = "VAE/tmp_eval/repeat_best"

    plot_csv(csv_dir)