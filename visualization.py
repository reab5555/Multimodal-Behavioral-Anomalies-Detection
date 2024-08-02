import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from utils import seconds_to_timecode
from anomaly_detection import determine_anomalies
import librosa
import librosa.display
import os


def plot_mse(df, mse_values, title, color='navy', time_threshold=3, anomaly_threshold=4):
    plt.figure(figsize=(16, 8), dpi=300)
    fig, ax = plt.subplots(figsize=(16, 8))

    if 'Seconds' not in df.columns:
        df['Seconds'] = df['Timecode'].apply(
            lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))

    # Ensure df and mse_values have the same length and remove NaN values
    min_length = min(len(df), len(mse_values))
    df = df.iloc[:min_length].copy()
    mse_values = mse_values[:min_length]

    # Remove NaN values and create a mask for valid data
    valid_mask = ~np.isnan(mse_values)
    df = df[valid_mask]
    mse_values = mse_values[valid_mask]

    # Function to identify continuous segments
    def get_continuous_segments(seconds, values, max_gap=1):
        segments = []
        current_segment = []
        for i, (sec, val) in enumerate(zip(seconds, values)):
            if not current_segment or (sec - current_segment[-1][0] <= max_gap):
                current_segment.append((sec, val))
            else:
                segments.append(current_segment)
                current_segment = [(sec, val)]
        if current_segment:
            segments.append(current_segment)
        return segments

    # Get continuous segments
    segments = get_continuous_segments(df['Seconds'], mse_values)

    # Plot each segment separately
    for segment in segments:
        segment_seconds, segment_mse = zip(*segment)
        ax.scatter(segment_seconds, segment_mse, color=color, alpha=0.3, s=5)

        # Calculate and plot rolling mean and std for this segment
        if len(segment) > 1:  # Only if there's more than one point in the segment
            segment_df = pd.DataFrame({'Seconds': segment_seconds, 'MSE': segment_mse})
            segment_df = segment_df.sort_values('Seconds')
            mean = segment_df['MSE'].rolling(window=min(10, len(segment)), min_periods=1, center=True).mean()
            std = segment_df['MSE'].rolling(window=min(10, len(segment)), min_periods=1, center=True).std()

            ax.plot(segment_df['Seconds'], mean, color=color, linewidth=0.5)
            ax.fill_between(segment_df['Seconds'], mean - std, mean + std, color=color, alpha=0.1)

    median = np.median(mse_values)
    ax.axhline(y=median, color='black', linestyle='--', label='Median Baseline')

    threshold = np.mean(mse_values) + anomaly_threshold * np.std(mse_values)
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Anomaly Threshold')
    ax.text(ax.get_xlim()[1], threshold, f'Anomaly Threshold', verticalalignment='center', horizontalalignment='left',
            color='red')

    anomalies = determine_anomalies(mse_values, anomaly_threshold)
    anomaly_frames = df['Frame'].iloc[anomalies].tolist()

    ax.scatter(df['Seconds'].iloc[anomalies], mse_values[anomalies], color='red', s=20, zorder=5)

    anomaly_data = list(zip(df['Timecode'].iloc[anomalies],
                            df['Seconds'].iloc[anomalies],
                            mse_values[anomalies]))
    anomaly_data.sort(key=lambda x: x[1])

    max_seconds = df['Seconds'].max()
    num_ticks = 80
    tick_locations = np.linspace(0, max_seconds, num_ticks)
    tick_labels = [seconds_to_timecode(int(s)) for s in tick_locations]

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=6)

    ax.set_xlabel('Timecode')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.close()
    return fig, anomaly_frames


def plot_mse_histogram(mse_values, title, anomaly_threshold, color='blue'):
    plt.figure(figsize=(16, 3), dpi=300)
    fig, ax = plt.subplots(figsize=(16, 3))

    ax.hist(mse_values, bins=100, edgecolor='black', color=color, alpha=0.7)
    ax.set_xlabel('Mean Squared Error')
    ax.set_ylabel('Number of Frames')
    ax.set_title(title)

    mean = np.mean(mse_values)
    std = np.std(mse_values)
    threshold = mean + anomaly_threshold * std

    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.close()
    return fig


def plot_mse_heatmap(mse_values, title, df):
    plt.figure(figsize=(20, 3), dpi=300)
    fig, ax = plt.subplots(figsize=(20, 3))

    # Reshape MSE values to 2D array for heatmap
    mse_2d = mse_values.reshape(1, -1)

    # Create heatmap
    sns.heatmap(mse_2d, cmap='YlOrRd', cbar=False, ax=ax)

    # Set x-axis ticks to timecodes
    num_ticks = min(60, len(mse_values))
    tick_locations = np.linspace(0, len(mse_values) - 1, num_ticks).astype(int)

    # Ensure tick_locations are within bounds
    tick_locations = tick_locations[tick_locations < len(df)]

    tick_labels = [df['Timecode'].iloc[i] if i < len(df) else '' for i in tick_locations]

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=90, ha='center', va='top')

    ax.set_title(title)

    # Remove y-axis labels
    ax.set_yticks([])

    plt.tight_layout()
    plt.close()
    return fig


def plot_posture(df, posture_scores, color='blue', anomaly_threshold=3):
    plt.figure(figsize=(16, 8), dpi=300)
    fig, ax = plt.subplots(figsize=(16, 8))

    df['Seconds'] = df['Timecode'].apply(
        lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))

    posture_data = [(frame, score) for frame, score in posture_scores.items() if score is not None]
    posture_frames, posture_scores = zip(*posture_data)

    # Create a new dataframe for posture data
    posture_df = pd.DataFrame({'Frame': posture_frames, 'Score': posture_scores})

    posture_df = posture_df.merge(df[['Frame', 'Seconds']], on='Frame', how='inner')

    ax.scatter(posture_df['Seconds'], posture_df['Score'], color=color, alpha=0.3, s=5)
    mean = posture_df['Score'].rolling(window=10).mean()
    ax.plot(posture_df['Seconds'], mean, color=color, linewidth=0.5)

    ax.set_xlabel('Timecode')
    ax.set_ylabel('Posture Score')
    ax.set_title("Body Posture Over Time")

    ax.grid(True, linestyle='--', alpha=0.7)

    max_seconds = df['Seconds'].max()
    num_ticks = 80
    tick_locations = np.linspace(0, max_seconds, num_ticks)
    tick_labels = [seconds_to_timecode(int(s)) for s in tick_locations]

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=6)

    plt.tight_layout()
    plt.close()
    return fig


def plot_stacked_mse_heatmaps(mse_face, mse_posture, mse_voice, df, title="Combined MSE Heatmaps"):
    plt.figure(figsize=(20, 6), dpi=300)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1, 1.2], 'hspace': 0})

    # Face heatmap
    sns.heatmap(mse_face.reshape(1, -1), cmap='Reds', cbar=False, ax=ax1, xticklabels=False, yticklabels=False)
    ax1.set_ylabel('Face', rotation=0, ha='right', va='center')
    ax1.yaxis.set_label_coords(-0.01, 0.5)

    # Posture heatmap
    sns.heatmap(mse_posture.reshape(1, -1), cmap='Reds', cbar=False, ax=ax2, xticklabels=False, yticklabels=False)
    ax2.set_ylabel('Posture', rotation=0, ha='right', va='center')
    ax2.yaxis.set_label_coords(-0.01, 0.5)

    # Voice heatmap
    sns.heatmap(mse_voice.reshape(1, -1), cmap='Reds', cbar=False, ax=ax3, yticklabels=False)
    ax3.set_ylabel('Voice', rotation=0, ha='right', va='center')
    ax3.yaxis.set_label_coords(-0.01, 0.5)

    # Set x-axis ticks to timecodes for the bottom subplot
    num_ticks = min(60, len(mse_voice))
    tick_locations = np.linspace(0, len(mse_voice) - 1, num_ticks).astype(int)
    tick_labels = [df['Timecode'].iloc[i] if i < len(df) else '' for i in tick_locations]
    ax3.set_xticks(tick_locations)
    ax3.set_xticklabels(tick_labels, rotation=90, ha='center', va='top')

    # Remove spines
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.close()
    return fig


def plot_audio_waveform(audio_path, title="Audio Waveform"):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Create the plot
    plt.figure(figsize=(20, 4))
    librosa.display.waveshow(y, sr=sr)

    # Set the x-axis to display timecodes
    max_time = librosa.get_duration(y=y, sr=sr)
    x_ticks = np.arange(0, max_time, max_time / 10)  # 10 ticks
    x_labels = [f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}" for t in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    return plt.gcf()
