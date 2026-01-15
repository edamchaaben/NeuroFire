import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()

def plot(scores, losses, mean_scores):
    """
    Plots the training progress in real-time.
    Plots Score (Left Y-Axis) and Loss (Right Y-Axis).
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Create subplots
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Scores (Blue)
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Score', color='tab:blue')
    ax1.plot(scores, label='Score', color='tab:blue', alpha=0.3)
    ax1.plot(mean_scores, label='Avg Score (100)', color='darkblue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Plot Loss (Red) - Secondary Axis
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Loss', color='tab:red')
    # Loss is often noisy, smooth it a bit if possible or just scatter
    if len(losses) > 0:
        # Calculate moving average of loss for clarity
        window = max(1, len(losses)//20)
        smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        # Pad to match length
        padding = [smoothed_loss[0]] * (len(losses) - len(smoothed_loss))
        smoothed_loss = np.concatenate([padding, smoothed_loss])
        
        ax2.plot(smoothed_loss, label='Loss (Smoothed)', color='tab:red', linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Training: NeuroFire (Double DQN)')
    plt.show(block=False)
    plt.pause(.1)
