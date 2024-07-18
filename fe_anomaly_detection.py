import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from facenet_pytorch import InceptionResnetV1, MTCNN
import mediapipe as mp
from fer import FER
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from PIL import Image
import gradio as gr
import tempfile
import shutil


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400

# Initialize models and other global variables
device = 'cuda'

mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.999, 0.999, 0.999], min_face_size=100,
              selection_method='largest')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
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

    # Load the video clip
    clip = VideoFileClip(video_path)

    original_fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * original_fps)
    step = max(1, original_fps / desired_fps)
    total_frames_to_extract = int(total_frames / step)

    frame_count = 0
    for t in np.arange(0, duration, step / original_fps):
        # Get the frame at time t
        frame = clip.get_frame(t)

        # Convert the frame to PIL Image and save it
        img = Image.fromarray(frame)
        img.save(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"))

        frame_count += 1

        # Report progress
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
            # Detect faces in batch
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
                            embedding, emotion = get_face_embedding_and_emotion(aligned_face_resized)
                            embeddings_by_frame[frame_num] = embedding
                            emotions_by_frame[frame_num] = emotion

        progress((i + len(batch_files)) / frame_count,
                 f"Processing frames {i + 1} to {min(i + len(batch_files), frame_count)} of {frame_count}")

    return embeddings_by_frame, emotions_by_frame


def cluster_embeddings(embeddings):
    if len(embeddings) < 2:
        print("Not enough embeddings for clustering. Assigning all to one cluster.")
        return np.zeros(len(embeddings), dtype=int)
    n_clusters = min(3, len(embeddings))  # Use at most 3 clusters
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_scaled)
    return clusters


def organize_faces_by_person(embeddings_by_frame, clusters, aligned_faces_folder, organized_faces_folder):
    for (frame_num, embedding), cluster in zip(embeddings_by_frame.items(), clusters):
        person_folder = os.path.join(organized_faces_folder, f"person_{cluster}")
        os.makedirs(person_folder, exist_ok=True)
        src = os.path.join(aligned_faces_folder, f"frame_{frame_num}_face.jpg")
        dst = os.path.join(person_folder, f"frame_{frame_num}_face.jpg")
        shutil.copy(src, dst)


def save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps, original_fps, output_folder,
                            num_components, video_duration):
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

    reducer = umap.UMAP(n_components=num_components, random_state=1)
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

    for i in range(num_components):
        df_data[f'Comp {i + 1}'] = embeddings_reduced_normalized[:, i]

    for emotion in emotions:
        df_data[emotion] = [e[emotion] for e in emotions_data]

    df = pd.DataFrame(df_data)

    return df, largest_cluster


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
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


def lstm_anomaly_detection(X, feature_columns, num_anomalies=10, epochs=100, batch_size=64):
    device = 'cuda'

    X = torch.FloatTensor(X).to(device)

    # Ensure X is 3D (batch, sequence, features)
    if X.dim() == 2:
        X = X.unsqueeze(0)
    elif X.dim() == 1:
        X = X.unsqueeze(0).unsqueeze(2)
    elif X.dim() > 3:
        raise ValueError(f"Input X should be 1D, 2D or 3D, but got {X.dim()} dimensions")

    print(f"X shape after reshaping: {X.shape}")

    train_size = int(0.85 * X.shape[1])
    X_train, X_val = X[:, :train_size, :], X[:, train_size:, :]

    model = LSTMAutoencoder(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train)
        loss_train = criterion(output_train, X_train.squeeze(0))
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(X_val)
            loss_val = criterion(output_val, X_val.squeeze(0))

    model.eval()
    with torch.no_grad():
        reconstructed = model(X).squeeze(0).cpu().numpy()

    # Compute anomalies for all features
    mse_all = np.mean(np.power(X.squeeze(0).cpu().numpy() - reconstructed, 2), axis=1)
    top_indices_all = mse_all.argsort()[-num_anomalies:][::-1]
    anomalies_all = np.zeros(len(mse_all), dtype=bool)
    anomalies_all[top_indices_all] = True

    # Compute anomalies for components only
    component_columns = [col for col in feature_columns if col.startswith('Comp')]
    component_indices = [feature_columns.index(col) for col in component_columns]

    if len(component_indices) > 0:
        mse_comp = np.mean(
            np.power(X.squeeze(0).cpu().numpy()[:, component_indices] - reconstructed[:, component_indices], 2), axis=1)
    else:
        mse_comp = mse_all  # If no components, use all features

    top_indices_comp = mse_comp.argsort()[-num_anomalies:][::-1]
    anomalies_comp = np.zeros(len(mse_comp), dtype=bool)
    anomalies_comp[top_indices_comp] = True

    return (anomalies_all, mse_all, top_indices_all,
            anomalies_comp, mse_comp, top_indices_comp,
            model)


def plot_emotion(df, emotion, num_anomalies, color):
    plt.figure(figsize=(16, 8), dpi=400)  # Increase DPI for higher quality
    fig, ax = plt.subplots(figsize=(16, 8))

    # Convert timecodes to seconds for proper plotting
    df['Seconds'] = df['Timecode'].apply(
        lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Seconds': df['Seconds'],
        'Emotion Score': df[emotion]
    })

    # Plot using seaborn
    sns.lineplot(x='Seconds', y='Emotion Score', data=plot_df, ax=ax, color=color)

    # Highlight top anomalies
    top_indices = np.argsort(df[emotion].values)[-num_anomalies:][::-1]
    ax.scatter(df['Seconds'].iloc[top_indices], df[emotion].iloc[top_indices], color='red', s=50, zorder=5)

    # Set x-axis
    max_seconds = df['Seconds'].max()
    ax.set_xlim(0, max_seconds)
    num_ticks = 80  # Reduce number of ticks for emotion graphs
    ax.set_xticks(np.linspace(0, max_seconds, num_ticks))
    ax.set_xticklabels([f"{int(x // 60):02d}:{int(x % 60):02d}" for x in ax.get_xticks()], rotation=90, ha='right')

    ax.set_xlabel('Time')
    ax.set_ylabel(f'{emotion.capitalize()} Score')
    ax.set_title(f'{emotion.capitalize()} Scores Over Time (Top {num_anomalies} in Red)')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_anomaly_scores(df, anomaly_scores, top_indices, title):
    plt.figure(figsize=(16, 8), dpi=400)  # Increase DPI for higher quality
    fig, ax = plt.subplots(figsize=(16, 8))

    # Convert timecodes to seconds for proper plotting
    df['Seconds'] = df['Timecode'].apply(
        lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))

    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Seconds': df['Seconds'],
        'Anomaly Score': anomaly_scores
    })

    # Plot using seaborn
    sns.lineplot(x='Seconds', y='Anomaly Score', data=plot_df, ax=ax)

    # Highlight top anomalies
    ax.scatter(df['Seconds'].iloc[top_indices], anomaly_scores[top_indices], color='red', s=50, zorder=5)

    # Set x-axis
    max_seconds = df['Seconds'].max()
    ax.set_xlim(0, max_seconds)
    num_ticks = 80  # Increase number of ticks for anomaly score graphs
    ax.set_xticks(np.linspace(0, max_seconds, num_ticks))
    ax.set_xticklabels([f"{int(x // 60):02d}:{int(x % 60):02d}" for x in ax.get_xticks()], rotation=90, ha='right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Anomaly Scores Over Time ({title})')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def get_random_face_sample(organized_faces_folder, largest_cluster, output_folder):
    person_folder = os.path.join(organized_faces_folder, f"person_{largest_cluster}")
    face_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]
    if face_files:
        random_face = np.random.choice(face_files)
        face_path = os.path.join(person_folder, random_face)
        output_path = os.path.join(output_folder, "random_face_sample.jpg")

        # Read the image and resize it to be smaller
        face_img = cv2.imread(face_path)
        small_face = cv2.resize(face_img, (160, 160))  # Resize to NxN pixels
        cv2.imwrite(output_path, small_face)

        return output_path
    return None


def process_video(video_path, num_anomalies, num_components, desired_fps, batch_size, progress=gr.Progress()):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

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
        embeddings_by_frame, emotions_by_frame = process_frames(frames_folder, aligned_faces_folder, frame_count,
                                                                progress, batch_size)

        if not embeddings_by_frame:
            return "No faces were extracted from the video.", None, None, None, None, None, None

        progress(0.6, "Clustering embeddings")
        embeddings = list(embeddings_by_frame.values())
        clusters = cluster_embeddings(embeddings)

        progress(0.7, "Organizing faces")
        organize_faces_by_person(embeddings_by_frame, clusters, aligned_faces_folder, organized_faces_folder)

        progress(0.8, "Saving person data")
        df, largest_cluster = save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps,
                                                      original_fps, temp_dir, num_components, video_duration)

        progress(0.9, "Performing anomaly detection")
        feature_columns = [col for col in df.columns if
                           col not in ['Frame', 'Timecode', 'Time (Minutes)', 'Embedding_Index']]
        X = df[feature_columns].values
        print(f"Shape of input data: {X.shape}")
        print(f"Feature columns: {feature_columns}")
        try:
            anomalies_all, anomaly_scores_all, top_indices_all, anomalies_comp, anomaly_scores_comp, top_indices_comp, _ = lstm_anomaly_detection(
                X, feature_columns, num_anomalies=num_anomalies, batch_size=batch_size)
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"X shape: {X.shape}")
            print(f"X dtype: {X.dtype}")
            return f"Error in anomaly detection: {str(e)}", None, None, None, None, None, None

        progress(0.95, "Generating plots")
        try:
            anomaly_plot_all = plot_anomaly_scores(df, anomaly_scores_all, top_indices_all, "All Features")
            anomaly_plot_comp = plot_anomaly_scores(df, anomaly_scores_comp, top_indices_comp, "Components Only")
            emotion_plots = [
                plot_emotion(df, 'fear', num_anomalies, 'purple'),
                plot_emotion(df, 'sad', num_anomalies, 'green'),
                plot_emotion(df, 'angry', num_anomalies, 'orange'),
                plot_emotion(df, 'happy', num_anomalies, 'darkblue'),
                plot_emotion(df, 'surprise', num_anomalies, 'gold'),
                plot_emotion(df, 'neutral', num_anomalies, 'grey')
            ]
        except Exception as e:
            return f"Error generating plots: {str(e)}", None, None, None, None, None, None, None, None, None

        progress(1.0, "Preparing results")
        results = f"Top {num_anomalies} anomalies (All Features):\n"
        results += "\n".join([f"{score:.4f} at {timecode}" for score, timecode in
                              zip(anomaly_scores_all[top_indices_all], df['Timecode'].iloc[top_indices_all].values)])
        results += f"\n\nTop {num_anomalies} anomalies (Components Only):\n"
        results += "\n".join([f"{score:.4f} at {timecode}" for score, timecode in
                              zip(anomaly_scores_comp[top_indices_comp], df['Timecode'].iloc[top_indices_comp].values)])

        for emotion in ['fear', 'sad', 'angry', 'happy', 'surprise', 'neutral']:
            top_indices = np.argsort(df[emotion].values)[-num_anomalies:][::-1]
            results += f"\n\nTop {num_anomalies} {emotion.capitalize()} Scores:\n"
            results += "\n".join([f"{df[emotion].iloc[i]:.4f} at {df['Timecode'].iloc[i]}" for i in top_indices])

        # Get a random face sample
        face_sample = get_random_face_sample(organized_faces_folder, largest_cluster, output_folder)

        return (
            results,
            anomaly_plot_all,
            anomaly_plot_comp,
            *emotion_plots,
            face_sample
        )


# Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(),
        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Number of Anomalies"),
        gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Number of Components"),
        gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Desired FPS"),
        gr.Slider(minimum=1, maximum=64, step=1, value=8, label="Batch Size")
    ],
    outputs=[
        gr.Textbox(label="Anomaly Detection Results"),
        gr.Plot(label="Anomaly Scores (All Features)"),
        gr.Plot(label="Anomaly Scores (Components Only)"),
        gr.Plot(label="Fear Anomalies"),
        gr.Plot(label="Sad Anomalies"),
        gr.Plot(label="Angry Anomalies"),
        gr.Plot(label="Happy Anomalies"),
        gr.Plot(label="Surprise Anomalies"),
        gr.Plot(label="Neutral Anomalies"),
        gr.Image(type="filepath", label="Random Face Sample of Most Frequent Person"),
    ],
    title="Facial Expressions Anomaly Detection",
    description="""
    This application detects anomalies in facial expressions and emotions from a video input. 
    It focuses on the most frequently appearing person in the video for analysis.

    Adjust the parameters as needed:
    - Number of Anomalies: How many top anomalies or high intensities to highlight
    - Number of Components: Complexity of the facial expression model
    - Desired FPS: Frames per second to analyze (lower for faster processing)
    - Batch Size: Affects processing speed and memory usage
    """
)

if __name__ == "__main__":
    iface.launch()
