# Multimodal Behavioral Anomalies Detection

The purpose of this tool is to detect anomalies in facial expressions, body language, and voice over the timeline of a video.   

It extracts faces, postures, and voice features from video frames, detects unique facial features, body postures, and speaker embeddings, and analyzes them to identify anomalies using time series analysis, specifically utilizing a variational autoencoder (VAE) approach.   

## Applications

- Identify suspicious behavior in surveillance footage.
- Analyze micro-expressions.
- Monitor and assess emotional states in communications.
- Evaluate changes in vocal tone and speech patterns.
-
- ## Features

- **Face Extraction**: Extracts faces from video frames using the MTCNN model.
- **Feature Embeddings**: Extracts facial feature embeddings using the InceptionResnetV1 model.
- **Body Posture Analysis**: Evaluates body postures using MediaPipe Pose.
- **Voice Analysis**: Extracts and segment speaker embeddings from audio using PyAnnote.
- **Anomaly Detection**: Uses Variational Autoencoder (VAE) to detect anomalies in facial expressions, body postures, and voice features over time.
- **Visualization**: Represents changes in facial expressions, body postures, and vocal tone over time, marking anomaly key points.

<img src="appendix/Anomay Detection.png" width="1050" alt="alt text">

## InceptionResnetV1
The InceptionResnetV1 model is a deep convolutional neural network used for facial recognition and facial attribute extraction.

- **Accuracy and Reliability**: Pre-trained on the VGGFace2 dataset, it achieves high accuracy in recognizing and differentiating between faces.
- **Feature Richness**: The embeddings capture rich facial details, essential for recognizing subtle expressions and variations.
- **Global Recognition**: Widely adopted in various facial recognition applications, demonstrating reliability and robustness across different scenarios.

## MediaPipe Pose
MediaPipe Pose is a versatile machine learning library designed for high-accuracy real-time posture estimation. Mediapipe Pose uses a deep learning model to detect body landmarks and infer body posture.

- **Real-Time Performance**: Capable of processing video frames at real-time speeds, making it suitable for live video analysis.
- **Accuracy and Precision**: Detects 33 body landmarks, including important joints and key points, enabling detailed posture and movement analysis.
- **Integration**: Easily integrates with other machine learning frameworks and tools, enhancing its versatility for various applications.

## Voice Analysis
The voice analysis module involves extracting and analyzing vocal features using speaker diarization and embedding models to capture key characteristics of the speaker's voice.

### Using PyAnnote and Diarization
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


## Examples

### A Death Sentence Verdict

<img src="appendix/wade_wilson_2.jpg" width="250" alt="alt text">

Wade Wilson, a 30-year-old from Fort Myers, Florida, was convicted in June 2024 for the October 2019 murders of Kristine Melton and Diane Ruiz in Cape Coral. During the trial, Wilson was notably cold and calm, showing little emotion throughout the proceedings, which many found unsettling. The jury recommended the death penalty, with the final sentencing set for July 23, 2024.

<p align="left">
<img src="appendix/1.jpg" width="50" alt="alt text">
<img src="appendix/2.jpg" width="50" alt="alt text">
<img src="appendix/3.jpg" width="50" alt="alt text">
<img src="appendix/4.jpg" width="50" alt="alt text">
<img src="appendix/5.jpg" width="50" alt="alt text">
<img src="appendix/6.jpg" width="50" alt="alt text">
<p/>

Sources:
1. [Fox News](https://www.foxnews.com/us/florida-double-murderer-viral-smug-soulless-courtroom-demeanor)
2. [WINK News](https://winknews.com/2024/06/13/wade-wilsons-lack-emotion-double-murder-trial/)
3. [YouTube](https://www.youtube.com/watch?v=8j8psgKXmRg)

### Detected Anomalies (Facial Features)
<p align="left">
<img src="appendix/anomaly_scores_all_features_plot.png" width="250" alt="alt text">
<img src="appendix/anomaly_scores_components_plot.png" width="250" alt="alt text">
<img src="appendix/anomaly_scores_embeddings_plot.png" width="250" alt="alt text">
<p/>

### Detected Anomalies (Body Posture)
<p align="left">
<img src="appendix/posture_scores_plot.png" width="250" alt="alt text">
<img src="appendix/posture_mse_plot.png" width="250" alt="alt text">
<p/>

### President Clinton's Interview with Jim Lehrer regarding Monica Lewinsky

President Bill Clinton's interview with Jim Lehrer about his relationship with Monica Lewinsky is another prominent example for testing our algorithm. During this interview, Clinton was asked difficult questions, and his facial expressions and body language were scrutinized. Our anomaly detection algorithm can identify key points where he was visibly distressed or potentially lying, appearing as anomalies in the detected data.

<p align="left">
<img src="appendix/clinton_lehrer_1.jpg" width="250" alt="alt text">
<img src="appendix/clinton_lehrer_2.jpg" width="250" alt="alt text">
<p/>

#### Detected Anomalies (Facial Features and Body Posture)
<p align="left">
<img src="appendix/clinton_face_anomaly_scores_plot.png" width="250" alt="alt text">
<img src="appendix/clinton_body_posture_anomaly_scores_plot.png" width="250" alt="alt text">
<p/>

When President Clinton was directly questioned about his relationship with Monica Lewinsky, anomalies were detected in his facial expressions, suggesting discomfort or deception.

## Limitations

- **Evaluation Challenges**: Since this is an unsupervised method, there is no labeled data to compare against. This makes it difficult to quantitatively evaluate the accuracy or effectiveness of the anomaly detection.
- **Subjectivity**: The concept of what constitutes an "anomaly" can be subjective and context-dependent. This can lead to false positives or negatives depending on the situation.
- **Lighting and Resolution**: Variability in lighting conditions, camera resolution, and frame rate can affect the quality of detected features and postures, leading to inconsistent results.
- **Audio Quality**: Background noise, poor audio quality, and overlapping speech can affect the accuracy of speaker diarization and voice embeddings.
- **Generalization**: The model may not generalize well to all types of videos and contexts. For example, trained embeddings may work well for a specific demographic but poorly for another.
- **Computationally Intensive**: Real-time processing of high-resolution video frames can be computationally demanding, requiring significant hardware resources.

### Micro-Expressions and Frame Rate Limitations
Videos recorded at 10 or 20 fps are not suitable for reliably capturing micro-expressions due to their rapid onset and brief duration. Higher frame rates, typically above 100 fps, are essential to ensure that these fleeting expressions are accurately captured and analyzed.

## Conclusion
This tool offers solutions for detecting emotional, posture, and vocal anomalies in video-based facial expressions, body language, and speech, beneficial for both forensic analysis and HUMINT operations. However, users should be aware of its limitations and the challenges inherent in unsupervised anomaly detection methodologies. By leveraging advanced computer vision techniques and the power of autoencoders, it provides crucial insights into human behavior in a timely manner, but results should be interpreted with caution and, where possible, supplemented with additional context and expert analysis.
