import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from config import OUTPUT_FOLDER
import os

def plot_emotion(df, emotion):
    plt.figure(figsize=(16, 8))
    values = df[emotion].values
    bars = plt.bar(range(len(df)), values, width=0.8)

    # Get indices of top 10 highest values
    top_10_indices = np.argsort(values)[-10:]

    # Color the bars
    for i, bar in enumerate(bars):
        if i in top_10_indices:
            bar.set_color('red')

    plt.xlabel('Timecode')
    plt.ylabel(f'{emotion.capitalize()} Score')
    plt.title(f'{emotion.capitalize()} Scores Over Time')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=100))
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([df['Timecode'].iloc[int(tick)] if tick >= 0 and tick < len(df) else '' for tick in ticks])

    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{emotion}_scores_plot.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # Print timecodes for top 10 highest values
    top_10_timecodes = df['Timecode'].iloc[top_10_indices].values
    print(f"\nTimecodes for top 10 highest {emotion} scores:")
    for timecode in top_10_timecodes:
        print(timecode)

def plot_anomaly_scores(df, anomaly_scores, top_indices, title, filename):
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(df)), anomaly_scores, width=0.8)
    for i in top_indices:
        bars[i].set_color('red')

    plt.xlabel('Timecode')
    plt.ylabel('Anomaly Score')
    plt.title(title)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=100))
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([df['Timecode'].iloc[int(tick)] if tick >= 0 and tick < len(df) else '' for tick in ticks])

    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=400, bbox_inches='tight')
    plt.close()

def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'lstm_training_loss.png'))
    plt.close()