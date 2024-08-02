import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.cluster import DBSCAN
import os
import shutil
import mediapipe as mp
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(face_img):
    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255
    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.cpu().numpy().flatten()

def cluster_faces(embeddings):
    if len(embeddings) < 2:
        print("Not enough faces for clustering. Assigning all to one cluster.")
        return np.zeros(len(embeddings), dtype=int)

    X = np.stack(embeddings)
    dbscan = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
    clusters = dbscan.fit_predict(X)

    if np.all(clusters == -1):
        print("DBSCAN assigned all to noise. Considering as one cluster.")
        return np.zeros(len(embeddings), dtype=int)

    return clusters

def organize_faces_by_person(embeddings_by_frame, clusters, faces_folder, organized_faces_folder):
    for (frame_num, embedding), cluster in zip(embeddings_by_frame.items(), clusters):
        person_folder = os.path.join(organized_faces_folder, f"person_{cluster}")
        os.makedirs(person_folder, exist_ok=True)
        src = os.path.join(faces_folder, f"frame_{frame_num}_face.jpg")
        dst = os.path.join(person_folder, f"frame_{frame_num}_face.jpg")
        shutil.copy(src, dst)
