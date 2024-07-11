from video_processing import extract_and_align_faces_from_video
from clustering import cluster_embeddings
from data_preparation import save_person_data_to_csv
from anomaly_detection import lstm_anomaly_detection
from visualization import plot_emotion, plot_anomaly_scores, plot_training_loss
from config import *
import shutil
import numpy as np

def main():
    embeddings_by_frame, emotions_by_frame, desired_fps, original_fps = extract_and_align_faces_from_video()

    if embeddings_by_frame:
        embeddings = list(embeddings_by_frame.values())
        clusters = cluster_embeddings(embeddings)

        print(f"Number of clusters: {len(set(clusters))}")
        for cluster in set(clusters):
            print(f"Cluster {cluster}: {list(clusters).count(cluster)} frames")

        df, largest_cluster = save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps, original_fps)

        # Remove aligned faces folder
        shutil.rmtree(aligned_faces_folder)

        # Keep only the largest cluster in organized_faces_folder
        for cluster in os.listdir(organized_faces_folder):
            if cluster != f"person_{largest_cluster}":
                shutil.rmtree(os.path.join(organized_faces_folder, cluster))

        # Anomaly detection
        feature_columns = [col for col in df.columns if col not in ['Frame', 'Timecode', 'Time (Minutes)', 'Embedding_Index']]

        # Anomaly detection with all features
        print("\nPerforming LSTM-based anomaly detection with all features...")
        anomalies_all, anomaly_scores_all, top_indices_all, model_all, train_losses_all, val_losses_all = lstm_anomaly_detection(
            df[feature_columns].values, feature_columns, num_anomalies=NUM_ANOMALIES)
        df['Anomaly_Score_All'] = anomaly_scores_all
        df['LSTM_anomaly_All'] = 0
        df.loc[top_indices_all, 'LSTM_anomaly_All'] = 1

        # Anomaly detection with only components
        component_columns = [col for col in df.columns if col.startswith('Comp')]
        print("\nPerforming LSTM-based anomaly detection with only components...")
        anomalies_comp, anomaly_scores_comp, top_indices_comp, model_comp, train_losses_comp, val_losses_comp = lstm_anomaly_detection(
            df[component_columns].values, component_columns, num_anomalies=NUM_ANOMALIES)
        df['Anomaly_Score_Comp'] = anomaly_scores_comp
        df['LSTM_anomaly_Comp'] = 0
        df.loc[top_indices_comp, 'LSTM_anomaly_Comp'] = 1

        # Load embeddings and perform anomaly detection
        embeddings_array = np.load(os.path.join(OUTPUT_FOLDER, 'face_embeddings.npy'))
        print("\nPerforming LSTM-based anomaly detection with embeddings...")
        anomalies_emb, anomaly_scores_emb, top_indices_emb, model_emb, train_losses_emb, val_losses_emb = lstm_anomaly_detection(
            embeddings_array, list(range(embeddings_array.shape[1])), num_anomalies=NUM_ANOMALIES)
        df['Anomaly_Score_Emb'] = anomaly_scores_emb
        df['LSTM_anomaly_Emb'] = 0
        df.loc[top_indices_emb, 'LSTM_anomaly_Emb'] = 1

        # Plot anomaly scores
        plot_anomaly_scores(df, anomaly_scores_all, top_indices_all, 'Anomaly Scores Over Time (All Features)', 'anomaly_scores_all_features_plot.png')
        plot_anomaly_scores(df, anomaly_scores_comp, top_indices_comp, 'Anomaly Scores Over Time (Components Only)', 'anomaly_scores_components_plot.png')
        plot_anomaly_scores(df, anomaly_scores_emb, top_indices_emb, 'Anomaly Scores Over Time (Embeddings)', 'anomaly_scores_embeddings_plot.png')

        # Plot emotion bar plots
        for emotion in ['fear', 'sad', 'angry']:
            plot_emotion(df, emotion)

        # Plot training losses
        plot_training_loss(train_losses_all, val_losses_all)
        plot_training_loss(train_losses_comp, val_losses_comp)
        plot_training_loss(train_losses_emb, val_losses_emb)

        print(f"\nTop {NUM_ANOMALIES} LSTM anomaly scores (All Features):")
        print(anomaly_scores_all[top_indices_all])
        print(f"Timecodes of top {NUM_ANOMALIES} anomalies (All Features):")
        print(df['Timecode'].iloc[top_indices_all].values)

        print(f"\nTop {NUM_ANOMALIES} LSTM anomaly scores (Components Only):")
        print(anomaly_scores_comp[top_indices_comp])
        print(f"Timecodes of top {NUM_ANOMALIES} anomalies (Components Only):")
        print(df['Timecode'].iloc[top_indices_comp].values)

        print(f"\nTop {NUM_ANOMALIES} LSTM anomaly scores (Embeddings):")
        print(anomaly_scores_emb[top_indices_emb])
        print(f"Timecodes of top {NUM_ANOMALIES} anomalies (Embeddings):")
        print(df['Timecode'].iloc[top_indices_emb].values)

        # Save the results
        print("\nSaving results...")
        output_file = os.path.join(OUTPUT_FOLDER, f'anomaly_detection_results.csv')
        df.to_csv(output_file, index=False)

        print(f"Anomaly detection completed. Results saved to '{output_file}'.")
        print(f"Anomaly scores plots and emotion plots saved to '{OUTPUT_FOLDER}'.")
    else:
        print("No faces were extracted from the video.")

if __name__ == "__main__":
    main()