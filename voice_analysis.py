import moviepy.editor as mp
from pyannote.audio import Pipeline
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio import Model
import os
import numpy as np


def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    video.audio.write_audiofile(audio_path)
    return audio_path


def diarize_speakers(audio_path):
    hf_token = os.environ.get("py_annote_hf_token")

    if not hf_token:
        raise ValueError(
            "py_annote_hf_token environment variable is not set. Please check your Hugging Face Space's Variables and secrets section.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(audio_path)

    # Identify the speakers and their segments
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))

        # Print each voice segment
        print(f"Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    print("\nSpeaker Summary:")
    for speaker, segments in speaker_segments.items():
        total_duration = sum(end - start for start, end in segments)
        print(f"Speaker {speaker}: Total duration = {total_duration:.2f}s")

    most_frequent_speaker = max(speaker_segments, key=lambda k: sum(end - start for start, end in speaker_segments[k]))
    print(f"\nMost frequent speaker: {most_frequent_speaker}")

    return diarization, most_frequent_speaker


def get_speaker_embeddings(audio_path, diarization, most_frequent_speaker, model_name="pyannote/embedding"):
    model = Model.from_pretrained(model_name, use_auth_token=os.environ.get("py_annote_hf_token"))
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sample_rate

    # Convert stereo to mono if necessary
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Minimum segment duration (in seconds)
    min_segment_duration = 0.5
    min_segment_length = int(min_segment_duration * sample_rate)

    embeddings = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker != most_frequent_speaker:
            continue

        start_frame = int(turn.start * sample_rate)
        end_frame = int(turn.end * sample_rate)
        segment = waveform[:, start_frame:end_frame]

        if segment.shape[1] > 0:
            # Pad short segments
            if segment.shape[1] < min_segment_length:
                padding = torch.zeros(1, min_segment_length - segment.shape[1])
                segment = torch.cat([segment, padding], dim=1)

            # Split long segments
            for i in range(0, segment.shape[1], min_segment_length):
                sub_segment = segment[:, i:i + min_segment_length]
                if sub_segment.shape[1] < min_segment_length:
                    padding = torch.zeros(1, min_segment_length - sub_segment.shape[1])
                    sub_segment = torch.cat([sub_segment, padding], dim=1)

                # Ensure the segment is on the correct device
                sub_segment = sub_segment.to(model.device)

                with torch.no_grad():
                    embedding = model(sub_segment)
                embeddings.append({
                    "time": turn.start + i / sample_rate,
                    "duration": min_segment_duration,
                    "embedding": embedding.cpu().numpy(),
                    "speaker": speaker
                })

    # Sort embeddings by time
    embeddings.sort(key=lambda x: x['time'])

    return embeddings, duration


def align_voice_embeddings(voice_embeddings, frame_count, fps, audio_duration):
    aligned_embeddings = []
    current_embedding_index = 0

    for frame in range(frame_count):
        frame_time = frame / fps

        # Find the correct embedding for the current frame time
        while (current_embedding_index < len(voice_embeddings) - 1 and
               voice_embeddings[current_embedding_index + 1]["time"] <= frame_time):
            current_embedding_index += 1

        current_embedding = voice_embeddings[current_embedding_index]

        # Check if the current frame is within the most frequent speaker's time range
        if current_embedding["time"] <= frame_time < (current_embedding["time"] + current_embedding["duration"]):
            aligned_embeddings.append(current_embedding["embedding"].flatten())
        else:
            # If not in the speaker's range, append a zero vector
            aligned_embeddings.append(np.zeros_like(voice_embeddings[0]["embedding"].flatten()))

    return aligned_embeddings