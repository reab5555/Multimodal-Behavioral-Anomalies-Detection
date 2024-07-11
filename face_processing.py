import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import mediapipe as mp
from fer import FER
from config import device

# Initialize MTCNN
mtcnn = MTCNN(keep_all=False, device=device, thresholds=[0.999, 0.999, 0.999], min_face_size=100,
              selection_method='largest')

# Initialize VGG-Face model (InceptionResnetV1 pre-trained on VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize the pre-trained emotion detector
emotion_detector = FER(mtcnn=False)

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