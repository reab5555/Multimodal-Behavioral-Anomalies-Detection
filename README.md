# Anomaly Detection in Facial Expressions and Body Language

This repository contains an advanced algorithm for detecting anomalies in facial expressions and body language over the timeline of a video. The tool extracts faces and postures from video frames, detects unique facial features and body postures, and analyzes them to identify anomalies using time series analysis, specifically utilizing a variational autoencoder (VAE) approach.

## Practical Applications

### Forensic Analysis
- Identify suspicious behavior in surveillance footage.
- Detect stress or duress in interrogation videos.

### Human Intelligence (HUMINT)
- Analyze micro-expressions.
- Monitor and assess emotional states in communications.

## Key Features

- **Face Extraction**: Extracts faces from video frames using the MTCNN model.
- **Feature Embeddings**: Extracts facial feature embeddings using the InceptionResnetV1 model.
- **Body Posture Analysis**: Evaluates body postures using MediaPipe Pose.
- **Anomaly Detection**: Uses Variational Autoencoder (VAE) to detect anomalies in facial expressions and body postures over time.

<img src="appendix/diagram.svg" width="1050" alt="alt text">

## Micro-Expressions
Paul Ekman’s work on facial expressions of emotion identified universal micro-expressions that reveal true emotions. These fleeting expressions, lasting only milliseconds, are challenging to detect but can be captured and analyzed using computer vision algorithms when analyzing frame-by-frame.

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

## Variational Autoencoder (VAE)
A Variational Autoencoder (VAE) is a type of neural network that learns to encode input data (like facial embeddings or posture scores) into a latent space and then reconstructs the data from this latent representation. VAEs not only learn to compress data but also to generate new data, making them particularly useful for anomaly detection.

- **Probabilistic Nature**: VAEs introduce a probabilistic approach to encoding, where the encoded representations are not single fixed points but distributions. This allows the model to learn a more robust representation of the data.
- **Reconstruction and Generation**: By comparing the reconstructed data to the original, VAEs can measure reconstruction errors. High errors indicate anomalies, as such data points do not conform well to the learned normal patterns.

## Setup Parameters
- **Frames Per Second (FPS)**: Frames per second to analyze (lower for faster processing).
- **Anomaly Detection Threshold**: Threshold for detecting anomalies (Standard Deviation).
  
## Example Scenario

### An Example from a Death Sentence Verdict

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

### Unsupervised Methodology

- **Evaluation Challenges**: Since this is an unsupervised method, there is no labeled data to compare against. This makes it difficult to quantitatively evaluate the accuracy or effectiveness of the anomaly detection.
- **Subjectivity**: The concept of what constitutes an "anomaly" can be subjective and context-dependent. This can lead to false positives or negatives depending on the situation.

### Data Quality and Quantity

- **Lighting and Resolution**: Variability in lighting conditions, camera resolution, and frame rate can affect the quality of detected features and postures, leading to inconsistent results.
- **Occlusions**: Faces and body parts that are partially obstructed or occluded can result in inaccurate embeddings and postural data.

### Model Limitations

- **Generalization**: The model may not generalize well to all types of videos and contexts. For example, trained embeddings may work well for a specific demographic but poorly for another.
- **Computationally Intensive**: Real-time processing of high-resolution video frames can be computationally demanding, requiring significant hardware resources.

## Conclusion
This tool offers solutions for detecting emotional and posture anomalies in video-based facial expressions and body language, beneficial for both forensic analysis and HUMINT operations. However, users should be aware of its limitations and the challenges inherent in unsupervised anomaly detection methodologies. By leveraging advanced computer vision techniques and the power of autoencoders, it provides crucial insights into human behavior in a timely manner, but results should be interpreted with caution and, where possible, supplemented with additional context and expert analysis.
