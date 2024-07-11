import os

import cv2
from tqdm import tqdm
from face_processing import get_face_embedding_and_emotion, alignFace, mtcnn
from config import VIDEO_FILE_PATH, aligned_faces_folder, desired_fps

def frame_to_timecode(frame_num, original_fps, desired_fps):
    total_seconds = frame_num / original_fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def extract_and_align_faces_from_video():
    video = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_FILE_PATH}")
        return
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = video.get(cv2.CAP_PROP_FPS)
    if frame_count == 0:
        print(f"Error: Video file at {VIDEO_FILE_PATH} appears to be empty")
        return
    faces_extracted = 0
    frames_with_no_faces = 0
    embeddings_by_frame = {}
    emotions_by_frame = {}

    for frame_num in tqdm(range(0, frame_count, int(original_fps / desired_fps)), desc="Extracting and aligning faces from video"):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()
        if not ret or frame is None:
            print(f"Error: Could not read frame {frame_num}")
            continue
        try:
            boxes, probs = mtcnn.detect(frame)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                if probs[0] >= 0.99:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame[y1:y2, x1:x2]
                    aligned_face = alignFace(face)
                    if aligned_face is not None:
                        aligned_face_resized = cv2.resize(aligned_face, (160, 160))
                        output_path = os.path.join(aligned_faces_folder, f"frame_{frame_num}_face.jpg")
                        cv2.imwrite(output_path, aligned_face_resized)
                        faces_extracted += 1
                        embedding, emotion = get_face_embedding_and_emotion(aligned_face_resized)
                        embeddings_by_frame[frame_num] = embedding
                        emotions_by_frame[frame_num] = emotion
            else:
                frames_with_no_faces += 1
        except Exception as e:
            print(f"Error processing frame {frame_num}: {str(e)}")
            frames_with_no_faces += 1
            continue

    video.release()
    print(f"Extracted and aligned {faces_extracted} faces from the video")
    print(f"Frames with no faces detected: {frames_with_no_faces}")
    print(f"Total frames with embeddings: {len(embeddings_by_frame)}")
    print(f"Total frames with emotions: {len(emotions_by_frame)}")
    return embeddings_by_frame, emotions_by_frame, desired_fps, original_fps