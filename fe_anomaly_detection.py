import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1, MTCNN
import mediapipe as mp
from fer import FER
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import umap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from PIL import Image
import gradio as gr
import tempfile
import shutil


matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400

# Initialize models and other global variables
device = 'cuda'

mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.98, 0.98, 0.98], min_face_size=100)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)
emotion_detector = FER(mtcnn=False)


def frame_to_timecode(frame_num, total_frames, duration):
    total_seconds = (frame_num / total_frames) * duration
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def get_face_embedding_and_emotion(face_img):
    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        embedding = model(face_tensor)

    emotions = emotion_detector.detect_emotions(face_img)
    if emotions:
        emotion_dict = emotions[0]['emotions']
    else:
        emotion_dict = {e: 0 for e in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}

    return embedding.cpu().numpy().flatten(), emotion_dict


def alignFace(img):
    img_raw = img.copy()
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    left_eye = np.array([[landmarks[33].x, landmarks[33].y], [landmarks[160].x, landmarks[160].y],
                         [landmarks[158].x, landmarks[158].y], [landmarks[144].x, landmarks[144].y],
                         [landmarks[153].x, landmarks[153].y], [landmarks[145].x, landmarks[145].y]])
    right_eye = np.array([[landmarks[362].x, landmarks[362].y], [landmarks[385].x, landmarks[385].y],
                          [landmarks[387].x, landmarks[387].y], [landmarks[263].x, landmarks[263].y],
                          [landmarks[373].x, landmarks[373].y], [landmarks[380].x, landmarks[380].y]])
    left_eye_center = left_eye.mean(axis=0).astype(np.int32)
    right_eye_center = right_eye.mean(axis=0).astype(np.int32)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    desired_angle = 0
    angle_diff = desired_angle - angle
    height, width = img_raw.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_diff, 1)
    new_img = cv2.warpAffine(img_raw, rotation_matrix, (width, height))
    return new_img


def extract_frames(video_path, output_folder, desired_fps, progress_callback=None):
    os.makedirs(output_folder, exist_ok=True)
    clip = VideoFileClip(video_path)
    original_fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * original_fps)
    step = max(1, original_fps / desired_fps)
    total_frames_to_extract = int(total_frames / step)

    frame_count = 0
    for t in np.arange(0, duration, step / original_fps):
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        img.save(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"))
        frame_count += 1
        if progress_callback:
            progress = min(100, (frame_count / total_frames_to_extract) * 100)
            progress_callback(progress, f"Extracting frame")
        if frame_count >= total_frames_to_extract:
            break
    clip.close()
    return frame_count, original_fps


def process_frames(frames_folder, aligned_faces_folder, frame_count, progress, batch_size):
    embeddings_by_frame = {}
    emotions_by_frame = {}
    aligned_face_paths = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i + batch_size]
        batch_frames = []
        batch_nums = []

        for frame_file in batch_files:
            frame_num = int(frame_file.split('_')[1].split('.')[0])
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                batch_frames.append(frame)
                batch_nums.append(frame_num)

        if batch_frames:
            batch_boxes, batch_probs = mtcnn.detect(batch_frames)

            for j, (frame, frame_num, boxes, probs) in enumerate(
                    zip(batch_frames, batch_nums, batch_boxes, batch_probs)):
                if boxes is not None and len(boxes) > 0 and probs[0] >= 0.99:
                    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        aligned_face = alignFace(face)
                        if aligned_face is not None:
                            aligned_face_resized = cv2.resize(aligned_face, (160, 160))
                            output_path = os.path.join(aligned_faces_folder, f"frame_{frame_num}_face.jpg")
                            cv2.imwrite(output_path, aligned_face_resized)
                            aligned_face_paths.append(output_path)
                            embedding, emotion = get_face_embedding_and_emotion(aligned_face_resized)
                            embeddings_by_frame[frame_num] = embedding
                            emotions_by_frame[frame_num] = emotion

        progress((i + len(batch_files)) / frame_count,
                 f"Processing frames {i + 1} to {min(i + len(batch_files), frame_count)} of {frame_count}")

    return embeddings_by_frame, emotions_by_frame, aligned_face_paths


def cluster_faces(embeddings):
    if len(embeddings) < 2:
        print("Not enough faces for clustering. Assigning all to one cluster.")
        return np.zeros(len(embeddings), dtype=int)

    X = np.stack(embeddings)

    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    clusters = dbscan.fit_predict(X)

    if np.all(clusters == -1):
        print("DBSCAN assigned all to noise. Considering as one cluster.")
        return np.zeros(len(embeddings), dtype=int)

    return clusters


def organize_faces_by_person(embeddings_by_frame, clusters, aligned_faces_folder, organized_faces_folder):
    for (frame_num, embedding), cluster in zip(embeddings_by_frame.items(), clusters):
        person_folder = os.path.join(organized_faces_folder, f"person_{cluster}")
        os.makedirs(person_folder, exist_ok=True)
        src = os.path.join(aligned_faces_folder, f"frame_{frame_num}_face.jpg")
        dst = os.path.join(person_folder, f"frame_{frame_num}_face.jpg")
        shutil.copy(src, dst)


def find_optimal_components(embeddings, max_components=20):
    pca = PCA(n_components=max_components)
    pca.fit(embeddings)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs. Number of Components')
    plt.grid(True)

    # Find elbow point
    differences = np.diff(cumulative_variance_ratio)
    elbow_point = np.argmin(differences) + 1

    plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow point: {elbow_point}')
    plt.legend()

    return elbow_point, plt


def save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps, original_fps, output_folder,
                            video_duration):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    person_data = {}

    for (frame_num, embedding), (_, emotion_dict), cluster in zip(embeddings_by_frame.items(),
                                                                  emotions_by_frame.items(), clusters):
        if cluster not in person_data:
            person_data[cluster] = []
        person_data[cluster].append((frame_num, embedding, {e: emotion_dict[e] for e in emotions}))

    largest_cluster = max(person_data, key=lambda k: len(person_data[k]))

    data = person_data[largest_cluster]
    data.sort(key=lambda x: x[0])
    frames, embeddings, emotions_data = zip(*data)

    embeddings_array = np.array(embeddings)
    np.save(os.path.join(output_folder, 'face_embeddings.npy'), embeddings_array)

    # Find optimal number of components
    optimal_components, _ = find_optimal_components(embeddings_array)

    reducer = umap.UMAP(n_components=optimal_components, random_state=1)
    embeddings_reduced = reducer.fit_transform(embeddings)

    scaler = MinMaxScaler(feature_range=(0, 1))
    embeddings_reduced_normalized = scaler.fit_transform(embeddings_reduced)

    total_frames = max(frames)
    timecodes = [frame_to_timecode(frame, total_frames, video_duration) for frame in frames]
    times_in_minutes = [frame / total_frames * video_duration / 60 for frame in frames]

    df_data = {
        'Frame': frames,
        'Timecode': timecodes,
        'Time (Minutes)': times_in_minutes,
        'Embedding_Index': range(len(embeddings))
    }

    # Add raw embeddings
    for i in range(len(embeddings[0])):
        df_data[f'Raw_Embedding_{i}'] = [embedding[i] for embedding in embeddings]

    for i in range(optimal_components):
        df_data[f'Comp {i + 1}'] = embeddings_reduced_normalized[:, i]

    for emotion in emotions:
        df_data[emotion] = [e[emotion] for e in emotions_data]

    df = pd.DataFrame(df_data)

    return df, largest_cluster


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        out = self.fc(outputs)
        return out


def lstm_anomaly_detection(X, feature_columns, raw_embedding_columns, epochs=100, batch_size=64):
    device = 'cuda'
    X = torch.FloatTensor(X).to(device)
    if X.dim() == 2:
        X = X.unsqueeze(0)
    elif X.dim() == 1:
        X = X.unsqueeze(0).unsqueeze(2)

    print(f"X shape after reshaping: {X.shape}")

    model = LSTMAutoencoder(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(X).squeeze(0).cpu().numpy()

    mse_all = np.mean(np.power(X.squeeze(0).cpu().numpy() - reconstructed, 2), axis=1)

    component_columns = [col for col in feature_columns if col.startswith('Comp')]
    component_indices = [feature_columns.index(col) for col in component_columns]

    if len(component_indices) > 0:
        mse_comp = np.mean(
            np.power(X.squeeze(0).cpu().numpy()[:, component_indices] - reconstructed[:, component_indices], 2), axis=1)
    else:
        mse_comp = mse_all

    raw_embedding_indices = [feature_columns.index(col) for col in raw_embedding_columns]
    mse_raw = np.mean(np.power(X.squeeze(0).cpu().numpy()[:, raw_embedding_indices] - reconstructed[:, raw_embedding_indices], 2), axis=1)

    return mse_all, mse_comp, mse_raw

def embedding_anomaly_detection(embeddings, epochs=100, batch_size=64):
    device = 'cuda'
    X = torch.FloatTensor(embeddings).to(device)
    if X.dim() == 2:
        X = X.unsqueeze(0)
    elif X.dim() == 1:
        X = X.unsqueeze(0).unsqueeze(2)

    model = LSTMAutoencoder(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X).squeeze(0).cpu().numpy()

    mse = np.mean(np.power(X.squeeze(0).cpu().numpy() - reconstructed, 2), axis=1)
    return mse

def determine_anomalies(mse_values, threshold=4):
    mean = np.mean(mse_values)
    std = np.std(mse_values)
    anomalies = mse_values > (mean + threshold * std)
    return anomalies


def plot_mse(df, mse_values, title, color='blue', time_threshold=1, hide_first_n=5):
    plt.figure(figsize=(16, 8), dpi=300)
    fig, ax = plt.subplots(figsize=(16, 8))

    df['Seconds'] = df['Timecode'].apply(
        lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))

    # Plot all points
    ax.scatter(df['Seconds'], mse_values, color=color, alpha=0.7, s=10)

    # Determine anomalies
    anomalies = determine_anomalies(mse_values)

    # Hide the first n anomalies
    visible_anomalies = np.where(anomalies)[0][hide_first_n:]
    ax.scatter(df['Seconds'].iloc[visible_anomalies], mse_values[visible_anomalies], color='red', s=50, zorder=5)

    # Group closely occurring anomalies and annotate only the highest MSE
    anomaly_data = list(zip(df['Timecode'].iloc[visible_anomalies],
                            df['Seconds'].iloc[visible_anomalies],
                            mse_values[visible_anomalies]))
    anomaly_data.sort(key=lambda x: x[1])  # Sort by seconds

    grouped_anomalies = []
    current_group = []
    for timecode, sec, mse in anomaly_data:
        if not current_group or sec - current_group[-1][1] <= time_threshold:
            current_group.append((timecode, sec, mse))
        else:
            grouped_anomalies.append(current_group)
            current_group = [(timecode, sec, mse)]
    if current_group:
        grouped_anomalies.append(current_group)

    for group in grouped_anomalies:
        highest_mse_anomaly = max(group, key=lambda x: x[2])
        timecode, sec, mse = highest_mse_anomaly
        ax.annotate(timecode, (sec, mse), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, color='red')

    # Add baseline (mean MSE) line
    mean_mse = np.mean(mse_values)
    ax.axhline(y=mean_mse, color='black', linestyle='--', linewidth=1)
    ax.text(df['Seconds'].max(), mean_mse, f'Baseline ({mean_mse:.6f})',
            verticalalignment='bottom', horizontalalignment='right', color='black', fontsize=8)

    # Set x-axis labels to timecodes
    max_seconds = df['Seconds'].max()
    num_ticks = 100
    tick_locations = np.linspace(0, max_seconds, num_ticks)
    tick_labels = [frame_to_timecode(int(s * df['Frame'].max() / max_seconds), df['Frame'].max(), max_seconds)
                   for s in tick_locations]

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=6)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close()
    return fig

def get_all_face_samples(organized_faces_folder, output_folder, largest_cluster):
    face_samples = {"most_frequent": [], "others": []}
    for cluster_folder in sorted(os.listdir(organized_faces_folder)):
        if cluster_folder.startswith("person_"):
            person_folder = os.path.join(organized_faces_folder, cluster_folder)
            face_files = sorted([f for f in os.listdir(person_folder) if f.endswith('.jpg')])
            if face_files:
                cluster_id = int(cluster_folder.split('_')[1])
                if cluster_id == largest_cluster:
                    for i, sample in enumerate(face_files):
                        face_path = os.path.join(person_folder, sample)
                        output_path = os.path.join(output_folder, f"face_sample_most_frequent_{i:04d}.jpg")
                        face_img = cv2.imread(face_path)
                        if face_img is not None:
                            small_face = cv2.resize(face_img, (160, 160))
                            cv2.imwrite(output_path, small_face)
                            face_samples["most_frequent"].append(output_path)
                else:
                    for i, sample in enumerate(face_files):
                        face_path = os.path.join(person_folder, sample)
                        output_path = os.path.join(output_folder, f"face_sample_other_{cluster_id:02d}_{i:04d}.jpg")
                        face_img = cv2.imread(face_path)
                        if face_img is not None:
                            small_face = cv2.resize(face_img, (160, 160))
                            cv2.imwrite(output_path, small_face)
                            face_samples["others"].append(output_path)
    return face_samples

def process_video(video_path, desired_fps, batch_size, progress=gr.Progress()):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize plot variables
    mse_plot_all = None
    mse_plot_comp = None
    mse_plot_raw = None
    emotion_plots = [None] * 6  # For the 6 emotions
    face_samples = {"most_frequent": [], "others": []}

    with tempfile.TemporaryDirectory() as temp_dir:
        aligned_faces_folder = os.path.join(temp_dir, 'aligned_faces')
        organized_faces_folder = os.path.join(temp_dir, 'organized_faces')
        os.makedirs(aligned_faces_folder, exist_ok=True)
        os.makedirs(organized_faces_folder, exist_ok=True)

        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()

        progress(0, "Starting frame extraction")
        frames_folder = os.path.join(temp_dir, 'extracted_frames')

        def extraction_progress(percent, message):
            progress(percent / 100, f"Extracting frames")

        frame_count, original_fps = extract_frames(video_path, frames_folder, desired_fps, extraction_progress)

        progress(1, "Frame extraction complete")
        progress(0.3, "Processing frames")
        embeddings_by_frame, emotions_by_frame, aligned_face_paths = process_frames(frames_folder, aligned_faces_folder,
                                                                                    frame_count,
                                                                                    progress, batch_size)

        if not aligned_face_paths:
            return ("No faces were extracted from the video.",
                    None, None, None, None, None, None, None, None, None, [], [])

        progress(0.6, "Clustering faces")
        embeddings = [embedding for _, embedding in embeddings_by_frame.items()]
        clusters = cluster_faces(embeddings)
        num_clusters = len(set(clusters))  # Get the number of unique clusters

        progress(0.7, "Organizing faces")
        organize_faces_by_person(embeddings_by_frame, clusters, aligned_faces_folder, organized_faces_folder)

        progress(0.8, "Saving person data")
        df, largest_cluster = save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps,
                                                      original_fps, temp_dir, video_duration)

        progress(0.85, "Getting face samples")
        face_samples = get_all_face_samples(organized_faces_folder, output_folder, largest_cluster)

        progress(0.9, "Performing anomaly detection")
        feature_columns = [col for col in df.columns if
                           col not in ['Frame', 'Timecode', 'Time (Minutes)', 'Embedding_Index']]
        raw_embedding_columns = [col for col in df.columns if col.startswith('Raw_Embedding_')]
        X = df[feature_columns].values

        try:
            mse_all, mse_comp, mse_raw = lstm_anomaly_detection(
                X, feature_columns, raw_embedding_columns, batch_size=batch_size)

            progress(0.95, "Generating plots")
            mse_plot_all = plot_mse(df, mse_all, "Facial Features + Emotions", color='blue', hide_first_n=5)
            mse_plot_comp = plot_mse(df, mse_comp, "Facial Features", color='deepskyblue', hide_first_n=5)
            mse_plot_raw = plot_mse(df, mse_raw, "Facial Embeddings", color='steelblue', hide_first_n=5)

            emotion_plots = [
                plot_mse(df, embedding_anomaly_detection(df[emotion].values.reshape(-1, 1)),
                         f"MSE: {emotion.capitalize()}", color=color, hide_first_n=5)
                for emotion, color in zip(['fear', 'sad', 'angry', 'happy', 'surprise', 'neutral'],
                                          ['purple', 'green', 'orange', 'darkblue', 'gold', 'grey'])
            ]

        except Exception as e:
            print(f"Error details: {str(e)}")
            return (f"Error in anomaly detection: {str(e)}",
                    None, None, None, None, None, None, None, None, None, [], [])

        progress(1.0, "Preparing results")
        results = f"Number of persons/clusters detected: {num_clusters}\n\n"
        results += f"Breakdown of persons/clusters:\n"
        for cluster_id in range(num_clusters):
            results += f"Person/Cluster {cluster_id + 1}: {len([c for c in clusters if c == cluster_id])} frames\n"

        return (
            results,
            mse_plot_all,
            mse_plot_comp,
            mse_plot_raw,
            *emotion_plots,
            face_samples["most_frequent"],
            face_samples["others"]
        )

# Define gallery outputs
gallery_outputs = [
    gr.Gallery(label="Most Frequent Person Random Samples", columns=5, rows=2, height="auto"),
    gr.Gallery(label="Other Persons Random Samples", columns=5, rows=1, height="auto")
]

# Update the Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(),
        gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Desired FPS"),
        gr.Slider(minimum=1, maximum=32, step=1, value=16, label="Batch Size")
    ],
    outputs=[
        gr.Textbox(label="Anomaly Detection Results"),
        gr.Plot(label="MSE: Facial Features + Emotions"),
        gr.Plot(label="MSE: Facial Features"),
        gr.Plot(label="MSE: Facial Embeddings"),
        gr.Plot(label="MSE: Fear"),
        gr.Plot(label="MSE: Sad"),
        gr.Plot(label="MSE: Angry"),
        gr.Plot(label="MSE: Happy"),
        gr.Plot(label="MSE: Surprise"),
        gr.Plot(label="MSE: Neutral"),
    ] + gallery_outputs,
    title="Facial Expressions Anomaly Detection",
    description="""
        This application detects anomalies in facial expressions and emotions from a video input. 
        It identifies distinct persons in the video and provides sample faces for each, with multiple samples for the most frequent person.

        The graphs show Mean Squared Error (MSE) values for different aspects of facial expressions and emotions over time.
        Each point represents a frame, with red points indicating detected anomalies.
        Anomalies are annotated with their corresponding timecodes.
        Higher MSE values indicate more unusual or anomalous expressions or emotions at that point in the video.

        Adjust the parameters as needed:
        - Desired FPS: Frames per second to analyze (lower for faster processing)
        - Batch Size: Affects processing speed and GPU memory usage
        """,
    allow_flagging="never"
)

# Launch the interface
iface.launch()