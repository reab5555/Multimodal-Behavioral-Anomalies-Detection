import math
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_posture_score(frame):
    image_height, image_width, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark

    # Use only body landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Calculate angles
    shoulder_angle = abs(math.degrees(math.atan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x)))
    hip_angle = abs(math.degrees(math.atan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x)))
    knee_angle = abs(math.degrees(math.atan2(right_knee.y - left_knee.y, right_knee.x - left_knee.x)))

    # Calculate vertical alignment
    shoulder_hip_alignment = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
    hip_knee_alignment = abs((left_hip.y + right_hip.y) / 2 - (left_knee.y + right_knee.y) / 2)
    # Add head landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    # Calculate head tilt
    head_tilt = abs(math.degrees(math.atan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x)))
    # Calculate head position relative to shoulders
    head_position = abs((nose.y - (left_shoulder.y + right_shoulder.y) / 2) /
                        ((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2))

    # Combine metrics into a single posture score (you may need to adjust these weights)
    posture_score = (
        (1 - abs(shoulder_angle - hip_angle) / 90) * 0.3 +
        (1 - abs(hip_angle - knee_angle) / 90) * 0.2 +
        (1 - shoulder_hip_alignment) * 0.1 +
        (1 - hip_knee_alignment) * 0.1 +
        (1 - abs(head_tilt - 90) / 90) * 0.15 +
        (1 - head_position) * 0.15
    )

    return posture_score, results.pose_landmarks

def draw_pose_landmarks(frame, landmarks):
    annotated_frame = frame.copy()
    # Include relevant landmarks for head position and body
    body_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]

    # Connections for head position and body
    body_connections = [
        (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    ]

    # Draw landmarks
    for landmark in body_landmarks:
        if landmark in landmarks.landmark:
            lm = landmarks.landmark[landmark]
            h, w, _ = annotated_frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_frame, (cx, cy), 5, (245, 117, 66), -1)

    # Draw connections
    for connection in body_connections:
        start_lm = landmarks.landmark[connection[0]]
        end_lm = landmarks.landmark[connection[1]]
        h, w, _ = annotated_frame.shape
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(annotated_frame, start_point, end_point, (245, 66, 230), 2)

    # Highlight head tilt
    left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    h, w, _ = annotated_frame.shape
    left_ear_point = (int(left_ear.x * w), int(left_ear.y * h))
    right_ear_point = (int(right_ear.x * w), int(right_ear.y * h))
    nose_point = (int(nose.x * w), int(nose.y * h))

    # Draw a line between ears to show head tilt
    cv2.line(annotated_frame, left_ear_point, right_ear_point, (0, 255, 0), 2)

    # Draw a line from nose to the midpoint between shoulders to show head forward/backward tilt
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_mid_point = (int(shoulder_mid_x * w), int(shoulder_mid_y * h))
    cv2.line(annotated_frame, nose_point, shoulder_mid_point, (0, 255, 0), 2)

    return annotated_frame