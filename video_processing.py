import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
import math
from face_analysis import get_face_embedding, cluster_faces, organize_faces_by_person
from pose_analysis import pose, calculate_posture_score, draw_pose_landmarks
from voice_analysis import get_speaker_embeddings, align_voice_embeddings, extract_audio_from_video, diarize_speakers
from anomaly_detection import anomaly_detection
from visualization import plot_mse, plot_mse_histogram, plot_mse_heatmap, plot_stacked_mse_heatmaps
from utils import frame_to_timecode
import pandas as pd
from facenet_pytorch import MTCNN
import torch
import mediapipe as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.98, 0.98, 0.98], min_face_size=200, post_process=False)


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
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_count += 1
        if progress_callback:
            progress = min(100, (frame_count / total_frames_to_extract) * 100)
            progress_callback(progress, f"Extracting frame")
        if frame_count >= total_frames_to_extract:
            break
    clip.close()
    return frame_count, original_fps


def is_frontal_face(face, landmarks):
    if landmarks is None:
        return False

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    nose_deviation = abs(nose[0] - eye_center[0]) / face.shape[1]

    return abs(eye_angle) < 10 and nose_deviation < 0.1


def process_frames(frames_folder, faces_folder, frame_count, progress):
    embeddings_by_frame = {}
    posture_scores_by_frame = {}
    posture_landmarks_by_frame = {}
    face_paths = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    for i, frame_file in enumerate(frame_files):
        frame_num = int(frame_file.split('_')[1].split('.')[0])
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

        if frame is not None:
            posture_score, posture_landmarks = calculate_posture_score(frame)
            posture_scores_by_frame[frame_num] = posture_score
            posture_landmarks_by_frame[frame_num] = posture_landmarks

            boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

            if boxes is not None and len(boxes) > 0 and probs[0] >= 0.99:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                face = frame[y1:y2, x1:x2]

                if face.size > 0 and is_frontal_face(face, landmarks[0]):
                    face_resized = cv2.resize(face, (160, 160))
                    output_path = os.path.join(faces_folder, f"frame_{frame_num}_face.jpg")
                    cv2.imwrite(output_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                    face_paths.append(output_path)
                    embedding = get_face_embedding(face_resized)
                    embeddings_by_frame[frame_num] = embedding

        progress((i + 1) / len(frame_files), f"Processing frame {i + 1} of {len(frame_files)}")

    return embeddings_by_frame, posture_scores_by_frame, posture_landmarks_by_frame, face_paths


def process_video(video_path, anomaly_threshold, desired_fps, progress=None):
    start_time = time.time()
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        faces_folder = os.path.join(temp_dir, 'faces')
        organized_faces_folder = os.path.join(temp_dir, 'organized_faces')
        os.makedirs(faces_folder, exist_ok=True)
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
        embeddings_by_frame, posture_scores_by_frame, posture_landmarks_by_frame, face_paths = process_frames(
            frames_folder, faces_folder,
            frame_count,
            progress)

        if not face_paths:
            raise ValueError("No faces were extracted from the video.")

        progress(0.6, "Clustering faces")
        embeddings = [embedding for _, embedding in embeddings_by_frame.items()]
        clusters = cluster_faces(embeddings)
        num_clusters = len(set(clusters))

        # Adding the 'Cluster' column to the DataFrame
        cluster_by_frame = {frame_num: cluster for frame_num, cluster in zip(embeddings_by_frame.keys(), clusters)}

        progress(0.65, "Organizing faces")
        organize_faces_by_person(embeddings_by_frame, clusters, faces_folder, organized_faces_folder)

        progress(0.7, "Saving person data")
        df, largest_cluster = save_person_data(embeddings_by_frame, clusters, desired_fps,
                                               original_fps, temp_dir, video_duration)

        df['Seconds'] = df['Timecode'].apply(
            lambda x: sum(float(t) * 60 ** i for i, t in enumerate(reversed(x.split(':')))))
        df['Cluster'] = df['Frame'].map(cluster_by_frame)

        progress(0.75, "Getting face samples")
        face_samples = get_all_face_samples(organized_faces_folder, output_folder, largest_cluster)

        progress(0.8, "Performing voice analysis")
        audio_path = extract_audio_from_video(video_path)
        diarization, most_frequent_speaker = diarize_speakers(audio_path)
        voice_embeddings, audio_duration = get_speaker_embeddings(audio_path, diarization, most_frequent_speaker)
        aligned_voice_embeddings = align_voice_embeddings(voice_embeddings, frame_count, original_fps, audio_duration)

        progress(0.85, "Performing anomaly detection")
        embedding_columns = [col for col in df.columns if col.startswith('Raw_Embedding_')]

        X_embeddings = df[embedding_columns].values
        X_posture = np.array([posture_scores_by_frame.get(frame, None) for frame in df['Frame']])
        X_posture = X_posture[X_posture != None].reshape(-1, 1)
        X_voice = np.array(aligned_voice_embeddings)

        if len(X_voice) > len(X_embeddings):
            X_voice = X_voice[:len(X_embeddings)]
        elif len(X_voice) < len(X_embeddings):
            padding = np.zeros((len(X_embeddings) - len(X_voice), X_voice.shape[1]))
            X_voice = np.vstack((X_voice, padding))

        try:
            if len(X_posture) == 0:
                raise ValueError("No valid posture data found")

            mse_embeddings, mse_posture, mse_voice = anomaly_detection(X_embeddings, X_posture, X_voice)

            progress(0.9, "Generating graphs")
            mse_plot_embeddings, anomaly_frames_embeddings = plot_mse(df, mse_embeddings, "Facial Features",
                                                                      color='navy',
                                                                      anomaly_threshold=anomaly_threshold)

            mse_histogram_embeddings = plot_mse_histogram(mse_embeddings, "MSE Distribution: Facial Features",
                                                          anomaly_threshold, color='navy')

            mse_plot_posture, anomaly_frames_posture = plot_mse(df, mse_posture, "Body Posture",
                                                                color='purple',
                                                                anomaly_threshold=anomaly_threshold)

            mse_histogram_posture = plot_mse_histogram(mse_posture, "MSE Distribution: Body Posture",
                                                       anomaly_threshold, color='purple')

            mse_plot_voice, anomaly_frames_voice = plot_mse(df, mse_voice, "Voice",
                                                            color='green',
                                                            anomaly_threshold=anomaly_threshold)

            mse_histogram_voice = plot_mse_histogram(mse_voice, "MSE Distribution: Voice",
                                                     anomaly_threshold, color='green')

            mse_heatmap_embeddings = plot_mse_heatmap(mse_embeddings, "Facial Features MSE Heatmap", df)
            mse_heatmap_posture = plot_mse_heatmap(mse_posture, "Body Posture MSE Heatmap", df)
            mse_heatmap_voice = plot_mse_heatmap(mse_voice, "Voice MSE Heatmap", df)

            stacked_heatmap = plot_stacked_mse_heatmaps(mse_embeddings, mse_posture, mse_voice, df,
                                                        "Combined MSE Heatmaps")

            progress(0.95, "Finishing generating graphs")

        except Exception as e:
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return (f"Error in video processing: {str(e)}",) + (None,) * 26

        progress(1.0, "Preparing results")
        results = f"Number of persons detected: {num_clusters}\n\n"
        results += "Breakdown:\n"
        for cluster_id in range(num_clusters):
            face_count = len([c for c in clusters if c == cluster_id])
            results += f"Person {cluster_id + 1}: {face_count} face frames\n"

        end_time = time.time()
        execution_time = end_time - start_time

        def add_timecode_to_image(image, timecode):
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()
            draw.text((10, 10), timecode, (255, 0, 0), font=font)
            return np.array(img_pil)

        anomaly_faces_embeddings = []
        for frame in anomaly_frames_embeddings:
            face_path = os.path.join(faces_folder, f"frame_{frame}_face.jpg")
            if os.path.exists(face_path):
                face_img = cv2.imread(face_path)
                if face_img is not None:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    timecode = df[df['Frame'] == frame]['Timecode'].iloc[0]
                    face_img_with_timecode = add_timecode_to_image(face_img, timecode)
                    anomaly_faces_embeddings.append(face_img_with_timecode)

        anomaly_frames_posture_images = []
        for frame in anomaly_frames_posture:
            frame_path = os.path.join(frames_folder, f"frame_{frame:04d}.jpg")
            if os.path.exists(frame_path):
                frame_img = cv2.imread(frame_path)
                if frame_img is not None:
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(frame_img)
                    if pose_results.pose_landmarks:
                        frame_img = draw_pose_landmarks(frame_img, pose_results.pose_landmarks)
                    timecode = df[df['Frame'] == frame]['Timecode'].iloc[0]
                    frame_img_with_timecode = add_timecode_to_image(frame_img, timecode)
                    anomaly_frames_posture_images.append(frame_img_with_timecode)

        return (
            execution_time,
            results,
            df,
            mse_embeddings,
            mse_posture,
            mse_voice,
            mse_plot_embeddings,
            mse_plot_posture,
            mse_plot_voice,
            mse_histogram_embeddings,
            mse_histogram_posture,
            mse_histogram_voice,
            mse_heatmap_embeddings,
            mse_heatmap_posture,
            mse_heatmap_voice,
            face_samples["most_frequent"],
            anomaly_faces_embeddings,
            anomaly_frames_posture_images,
            faces_folder,
            frames_folder,
            stacked_heatmap

        )


def save_person_data(embeddings_by_frame, clusters, desired_fps, original_fps, output_folder, video_duration):
    person_data = {}

    for (frame_num, embedding), cluster in zip(embeddings_by_frame.items(), clusters):
        if cluster not in person_data:
            person_data[cluster] = []
        person_data[cluster].append((frame_num, embedding))

    largest_cluster = max(person_data, key=lambda k: len(person_data[k]))

    data = person_data[largest_cluster]
    data.sort(key=lambda x: x[0])
    frames, embeddings = zip(*data)

    embeddings_array = np.array(embeddings)
    np.save(os.path.join(output_folder, 'face_embeddings.npy'), embeddings_array)

    total_frames = max(frames)
    timecodes = [frame_to_timecode(frame, total_frames, video_duration) for frame in frames]

    df_data = {
        'Frame': frames,
        'Timecode': timecodes,
        'Embedding_Index': range(len(embeddings))
    }

    for i in range(len(embeddings[0])):
        df_data[f'Raw_Embedding_{i}'] = [embedding[i] for embedding in embeddings]

    df = pd.DataFrame(df_data)

    return df, largest_cluster


def get_all_face_samples(organized_faces_folder, output_folder, largest_cluster, max_samples=200):
    face_samples = {"most_frequent": [], "others": []}
    for cluster_folder in sorted(os.listdir(organized_faces_folder)):
        if cluster_folder.startswith("person_"):
            person_folder = os.path.join(organized_faces_folder, cluster_folder)
            face_files = sorted([f for f in os.listdir(person_folder) if f.endswith('.jpg')])
            if face_files:
                cluster_id = int(cluster_folder.split('_')[1])
                if cluster_id == largest_cluster:
                    for i, sample in enumerate(face_files[:max_samples]):
                        face_path = os.path.join(person_folder, sample)
                        output_path = os.path.join(output_folder, f"face_sample_most_frequent_{i:04d}.jpg")
                        face_img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
                        if face_img is not None:
                            small_face = cv2.resize(face_img, (160, 160))
                            cv2.imwrite(output_path, cv2.cvtColor(small_face, cv2.COLOR_RGB2BGR))
                            face_samples["most_frequent"].append(output_path)
                        if len(face_samples["most_frequent"]) >= max_samples:
                            break
                else:
                    remaining_samples = max_samples - len(face_samples["others"])
                    if remaining_samples > 0:
                        for i, sample in enumerate(face_files[:remaining_samples]):
                            face_path = os.path.join(person_folder, sample)
                            output_path = os.path.join(output_folder, f"face_sample_other_{cluster_id:02d}_{i:04d}.jpg")
                            face_img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
                            if face_img is not None:
                                small_face = cv2.resize(face_img, (160, 160))
                                cv2.imwrite(output_path, cv2.cvtColor(small_face, cv2.COLOR_RGB2BGR))
                                face_samples["others"].append(output_path)
                            if len(face_samples["others"]) >= max_samples:
                                break
    return face_samples