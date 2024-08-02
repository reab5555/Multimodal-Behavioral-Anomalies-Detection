import gradio as gr
import time
from video_processing import process_video
from PIL import Image
import matplotlib
import numpy as np

matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400


def process_and_show_completion(video_input_path, anomaly_threshold_input, fps, progress=gr.Progress()):
    try:
        print("Starting video processing...")
        results = process_video(video_input_path, anomaly_threshold_input, fps, progress=progress)
        print("Video processing completed.")

        if isinstance(results[0], str) and results[0].startswith("Error"):
            print(f"Error occurred: {results[0]}")
            return [results[0]] + [None] * 25

        exec_time, results_summary, df, mse_embeddings, mse_posture, mse_voice, \
            mse_plot_embeddings, mse_plot_posture, mse_plot_voice, \
            mse_histogram_embeddings, mse_histogram_posture, mse_histogram_voice, \
            mse_heatmap_embeddings, mse_heatmap_posture, mse_heatmap_voice, \
            face_samples_frequent, \
            anomaly_faces_embeddings, anomaly_frames_posture_images, \
            faces_folder, frames_folder, \
            stacked_heatmap = results

        anomaly_faces_embeddings_pil = [Image.fromarray(face) for face in
                                        anomaly_faces_embeddings] if anomaly_faces_embeddings is not None else []
        anomaly_frames_posture_pil = [Image.fromarray(frame) for frame in
                                      anomaly_frames_posture_images] if anomaly_frames_posture_images is not None else []

        face_samples_frequent = [Image.open(path) for path in
                                 face_samples_frequent] if face_samples_frequent is not None else []

        output = [
            exec_time, results_summary,
            df, mse_embeddings, mse_posture, mse_voice,
            mse_plot_embeddings, mse_plot_posture, mse_plot_voice,
            mse_histogram_embeddings, mse_histogram_posture, mse_histogram_voice,
            mse_heatmap_embeddings, mse_heatmap_posture, mse_heatmap_voice,
            anomaly_faces_embeddings_pil, anomaly_frames_posture_pil,
            face_samples_frequent,
            faces_folder, frames_folder,
            mse_embeddings, mse_posture, mse_voice,
            stacked_heatmap

        ]

        return output

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return [error_message] + [None] * 25


def show_results(outputs):
    return gr.Group(visible=True)


def set_video_path(video_path):
    return video_path


with gr.Blocks() as iface:
    gr.Markdown("""
    # Multimodal Behavioral Anomalies Detection

    This tool detects anomalies in facial expressions, body language, and voice over the timeline of a video.
    It extracts faces, postures, and voice from video frames, and analyzes them to identify anomalies using time series analysis and a variational autoencoder (VAE) approach.
    """)

    with gr.Row():
        video_input = gr.Video()

    anomaly_threshold = gr.Slider(minimum=1, maximum=5, step=0.1, value=4,
                                  label="Anomaly Detection Threshold (Standard deviation)")
    fps_slider = gr.Slider(minimum=5, maximum=20, step=1, value=10, label="Frames Per Second (FPS)")
    process_btn = gr.Button("Detect Anomalies")
    progress_bar = gr.Progress()
    execution_time = gr.Number(label="Execution Time (seconds)")

    with gr.Group(visible=False) as results_group:
        with gr.Tabs():
            with gr.TabItem("Facial Features"):
                results_text = gr.TextArea(label="Faces Breakdown", lines=5)
                mse_features_plot = gr.Plot(label="MSE: Facial Features")
                mse_features_hist = gr.Plot(label="MSE Distribution: Facial Features")
                mse_features_heatmap = gr.Plot(label="MSE Heatmap: Facial Features")
                anomaly_frames_features = gr.Gallery(label="Anomaly Frames (Facial Features)", columns=6, rows=2,
                                                     height="auto")
                face_samples_most_frequent = gr.Gallery(label="Most Frequent Person Samples", columns=10, rows=2,
                                                        height="auto")

            with gr.TabItem("Body Posture"):
                mse_posture_plot = gr.Plot(label="MSE: Body Posture")
                mse_posture_hist = gr.Plot(label="MSE Distribution: Body Posture")
                mse_posture_heatmap = gr.Plot(label="MSE Heatmap: Body Posture")
                anomaly_frames_posture = gr.Gallery(label="Anomaly Frames (Body Posture)", columns=6, rows=2,
                                                    height="auto")

            with gr.TabItem("Voice"):
                mse_voice_plot = gr.Plot(label="MSE: Voice")
                mse_voice_hist = gr.Plot(label="MSE Distribution: Voice")
                mse_voice_heatmap = gr.Plot(label="MSE Heatmap: Voice")

            with gr.TabItem("Combined"):
                stacked_heatmap_plot = gr.Plot(label="Combined MSE Heatmaps")

    with gr.Row():
        with gr.Group(visible=True) as results_group2:
            with gr.TabItem("Description"):
                example_btn_1 = gr.Button("Load Example: Bill Clinton and Jim Lehrer")
                example_btn_2 = gr.Button("Load Example: Wade Wilson Penalty Phase Trial")
                gr.HTML("<div style='height: 30px;'></div>")
                gr.Image(value="appendix/Anomay Detection.png", label='Flowchart')
                gr.HTML("<div style='height: 20px;'></div>")
                gr.Markdown("""

                            # Multimodal Behavioral Anomalies Detection

                            The purpose of this tool is to detect anomalies in facial expressions, body language, and voice over the timeline of a video.   

                            It extracts faces, postures, and voice features from video frames, detects unique facial features, body postures, and speaker embeddings, and analyzes them to identify anomalies using time series analysis, specifically utilizing a variational autoencoder (VAE) approach.   

                            ## Applications

                            - Identify suspicious behavior in surveillance footage.
                            - Analyze micro-expressions.
                            - Monitor and assess emotional states in communications.
                            - Evaluate changes in vocal tone and speech patterns.

                            ## Features

                            - **Face Extraction**: Extracts faces from video frames using the MTCNN model.
                            - **Feature Embeddings**: Extracts facial feature embeddings using the InceptionResnetV1 model.
                            - **Body Posture Analysis**: Evaluates body postures using MediaPipe Pose.
                            - **Voice Analysis**: Extracts and segment speaker embeddings from audio using PyAnnote.
                            - **Anomaly Detection**: Uses Variational Autoencoder (VAE) to detect anomalies in facial expressions, body postures, and voice features over time.
                            - **Visualization**: Represents changes in facial expressions, body postures, and vocal tone over time, marking anomaly key points.

                            ## InceptionResnetV1
                            The InceptionResnetV1 model is a deep convolutional neural network used for facial recognition and facial attribute extraction.

                            - **Accuracy and Reliability**: Pre-trained on the VGGFace2 dataset, it achieves high accuracy in recognizing and differentiating between faces.
                            - **Feature Richness**: The embeddings capture rich facial details, essential for recognizing subtle expressions and variations.
                            - **Global Recognition**: Widely adopted in various facial recognition applications, demonstrating reliability and robustness across different scenarios.

                            ## MediaPipe Pose
                            MediaPipe Pose is a versatile machine learning library designed for high-accuracy real-time posture estimation. Mediapipe Pose uses a deep learning model to detect body landmarks and infer body posture.

                            - **Real-Time Performance**: Capable of processing video frames at real-time speeds, making it suitable for live video analysis.
                            - **Accuracy and Precision**: Detects body landmarks, including important joints and key points, enabling detailed posture and movement analysis.
                            - **Integration**: Easily integrates with other machine learning frameworks and tools, enhancing its versatility for various applications.

                            ## Voice Analysis
                            The voice analysis module involves extracting and analyzing vocal features using speaker diarization and embedding models to capture key characteristics of the speaker's voice.

                            PyAnnote is a toolkit for speaker diarization and voice analysis.
                            - **Speaker Diarization**: Identifies voice segments and classifies them by speaker.
                            - **Speaker Embeddings**: Captures voice characteristics using a pre-trained embedding model.

                            ## Variational Autoencoder (VAE)
                            A Variational Autoencoder (VAE) is a type of neural network that learns to encode input data (like facial embeddings or posture scores) into a latent space and then reconstructs the data from this latent representation. VAEs not only learn to compress data but also to generate new data, making them particularly useful for anomaly detection.

                            - **Probabilistic Nature**: VAEs introduce a probabilistic approach to encoding, where the encoded representations are not single fixed points but distributions. This allows the model to learn a more robust representation of the data.
                            - **Reconstruction and Generation**: By comparing the reconstructed data to the original, VAEs can measure reconstruction errors. High errors indicate anomalies, as such data points do not conform well to the learned normal patterns.

                            ## Setup Parameters
                            - **Frames Per Second (FPS)**: Frames per second to analyze (lower for faster processing).
                            - **Anomaly Detection Threshold**: Threshold for detecting anomalies (Standard Deviation).

                            ## Micro-Expressions
                            Paul Ekman’s work on facial expressions of emotion identified universal micro-expressions that reveal true emotions. These fleeting expressions, lasting only milliseconds, are challenging to detect but can be captured and analyzed using computer vision algorithms when analyzing frame-by-frame.

                            ### Micro-Expressions and Frame Rate Analysis
                            Micro-expressions are brief, involuntary facial expressions that typically last between 1/25 to 1/5 of a second (40-200 milliseconds). To capture these fleeting expressions, a high frame rate is essential.

                            ### 10 fps

                            - **Frame Interval** Each frame is captured every 100 milliseconds.
                            - **Effectiveness** Given that micro-expressions can last as short as 40 milliseconds, a frame rate of 10 fps is insufficient. Many micro-expressions would begin and end between frames, making it highly likely that they would be missed entirely.

                            ### 20 fps

                            - **Frame Interval** Each frame is captured every 50 milliseconds.
                            - **Effectiveness** While 20 fps is better than 10 fps, it is still inadequate. Micro-expressions can still occur and resolve within the 50-millisecond interval between frames, leading to incomplete or missed captures.

                            ### High-Speed Cameras

                            Effective capture of micro-expressions generally requires frame rates above 100 fps. High-speed video systems designed for micro-expression detection often operate at 118 fps or higher, with some systems reaching up to 200 fps.

                            ## Limitations

                            - **Evaluation Challenges**: Since this is an unsupervised method, there is no labeled data to compare against. This makes it difficult to quantitatively evaluate the accuracy or effectiveness of the anomaly detection.
                            - **Subjectivity**: The concept of what constitutes an "anomaly" can be subjective and context-dependent. This can lead to false positives or negatives depending on the situation.
                            - **Lighting and Resolution**: Variability in lighting conditions, camera resolution, and frame rate can affect the quality of detected features and postures, leading to inconsistent results.
                            - **Audio Quality**: Background noise, poor audio quality, and overlapping speech can affect the accuracy of speaker diarization and voice embeddings.
                            - **Generalization**: The model may not generalize well to all types of videos and contexts. For example, trained embeddings may work well for a specific demographic but poorly for another.
                            - **Computationally Intensive**: Real-time processing of high-resolution video frames can be computationally demanding, requiring significant hardware resources.
                            - **Frame Rate Limitations**: Videos recorded at 10 or 20 fps are not suitable for reliably capturing micro-expressions due to their rapid onset and brief duration.


                        """)

    df_store = gr.State()
    mse_features_store = gr.State()
    mse_posture_store = gr.State()
    mse_voice_store = gr.State()
    faces_folder_store = gr.State()
    frames_folder_store = gr.State()
    mse_heatmap_embeddings_store = gr.State()
    mse_heatmap_posture_store = gr.State()
    mse_heatmap_voice_store = gr.State()

    process_btn.click(
        process_and_show_completion,
        inputs=[video_input, anomaly_threshold, fps_slider],
        outputs=[
            execution_time, results_text, df_store,
            mse_features_store, mse_posture_store, mse_voice_store,
            mse_features_plot, mse_posture_plot, mse_voice_plot,
            mse_features_hist, mse_posture_hist, mse_voice_hist,
            mse_features_heatmap, mse_posture_heatmap, mse_voice_heatmap,
            anomaly_frames_features, anomaly_frames_posture,
            face_samples_most_frequent,
            faces_folder_store, frames_folder_store,
            mse_heatmap_embeddings_store, mse_heatmap_posture_store, mse_heatmap_voice_store,
            stacked_heatmap_plot
        ]
    ).then(
        show_results,
        inputs=None,
        outputs=results_group
    )

    example_btn_1.click(
        lambda: "appendix/Bill Clinton and Jim Lehrer.mp4",
        inputs=[],
        outputs=video_input
    )

    example_btn_2.click(
        lambda: "appendix/Wade_Wilson_Penalty_Phase_Trial.mp4",
        inputs=[],
        outputs=video_input
    )

if __name__ == "__main__":
    iface.launch()