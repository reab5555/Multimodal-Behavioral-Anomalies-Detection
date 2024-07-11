import os

device = 'cuda'
NUM_ANOMALIES = 10
NUM_COMPONENTS = 3
desired_fps = 20
batch_size = 16
VIDEO_FILE_PATH = "vide_file_imput"
OUTPUT_FOLDER = os.path.dirname(VIDEO_FILE_PATH)

aligned_faces_folder = os.path.join(OUTPUT_FOLDER, 'aligned_faces')
organized_faces_folder = os.path.join(OUTPUT_FOLDER, 'organized_faces')

# Create necessary folders
for folder in [aligned_faces_folder, organized_faces_folder]:
    os.makedirs(folder, exist_ok=True)