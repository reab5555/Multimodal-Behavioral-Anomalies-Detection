import os
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler
from config import NUM_COMPONENTS, OUTPUT_FOLDER
from video_processing import frame_to_timecode

def save_person_data_to_csv(embeddings_by_frame, emotions_by_frame, clusters, desired_fps, original_fps):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']
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

    # Save embeddings as numpy array
    embeddings_array = np.array(embeddings)
    np.save(os.path.join(OUTPUT_FOLDER, 'face_embeddings.npy'), embeddings_array)

    reducer = umap.UMAP(n_components=NUM_COMPONENTS, random_state=1)
    embeddings_reduced = reducer.fit_transform(embeddings)

    scaler = MinMaxScaler(feature_range=(0, 1))
    embeddings_reduced_normalized = scaler.fit_transform(embeddings_reduced)

    timecodes = [frame_to_timecode(frame, original_fps, desired_fps) for frame in frames]
    times_in_minutes = [frame / (original_fps * 60) for frame in frames]

    df_data = {
        'Frame': frames,
        'Timecode': timecodes,
        'Time (Minutes)': times_in_minutes,
        'Embedding_Index': range(len(embeddings))
    }

    for i in range(NUM_COMPONENTS):
        df_data[f'Comp {i + 1}'] = embeddings_reduced_normalized[:, i]

    for emotion in emotions:
        df_data[emotion] = [e[emotion] for e in emotions_data]

    df = pd.DataFrame(df_data)

    return df, largest_cluster