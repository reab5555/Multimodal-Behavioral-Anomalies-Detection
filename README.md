<img src="appendix/icon.jpeg" width="100" alt="alt text">

# Facial-Expression-Anomaly-Detection

This repository contains an advanced algorithm for detecting anomalies in facial expressions over the timeline of a video using time series analysis, specifically utilizing an LSTM autoencoder. The tool extracts faces from video frames, detects unique facial features, and analyzes emotional facial expressions to identify anomalies.

## Practical Applications

### Forensic Analysis
- Identify suspicious behavior in surveillance footage.
- Detect stress or duress in interrogation videos.

### Human Intelligence (HUMINT)
- Analyze micro-expressions.
- Monitor and assess emotional states in communications.

## Key Features

- **Face Extraction**: Extracts faces from video frames using MTCNN model.
- **Face Alignment**: Aligns and normalizes faces using MediaPipe for consistent analysis.
- **Feature Embeddings**: Extracts facial feature embeddings using the InceptionResnetV1/VGG-Face model.
- **Emotion Detection**: Identifies facial expressions and categorizes emotions using the FER model.
- **Anomaly Detection**: Uses an LSTM autoencoder to detect anomalies in facial expressions over time.

## Micro-Expressions
Paul Ekman’s work on facial expressions of emotion identified universal micro-expressions that reveal true emotions. These fleeting expressions, which last only milliseconds, are incredibly difficult for humans to detect but can be captured and analyzed using computer vision algorithms.

## InceptionResnetV1
The InceptionResnetV1 model is a deep convolutional neural network widely used for facial recognition and facial attributes extraction.

- **Accuracy and Reliability**: Pre-trained on the VGGFace2 dataset, it achieves high accuracy in recognizing and differentiating between faces.
- **Feature Richness**: The embeddings capture rich facial details, essential for recognizing subtle expressions and variations.
- **Global Recognition**: Widely adopted in various facial recognition applications, demonstrating reliability and robustness across different scenarios.

## FER (Facial Expression Recognition)
The FER model used in our pipeline is a pre-trained neural network designed to identify emotional states from facial expressions.

- **Accuracy and Reliability**: Pre-trained on a large dataset of facial images labeled with emotional states, achieving high accuracy in identifying seven basic emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.
- **Robustness**: Capable of recognizing emotions in varying lighting conditions, facial orientations, and occlusions, making it highly reliable for practical applications.

## LSTM Autoencoder
An LSTM (Long Short-Term Memory) Autoencoder is a neural network designed for sequential data, consisting of an encoder that compresses input sequences into a fixed-length representation and a decoder that reconstructs the sequence from this representation.

### In Our Facial-Expression Anomaly Detection:
- **Input Preparation**: Facial embeddings are extracted from video frames.
- **Sequence Creation**: These embeddings form a chronological sequence.
- **Training**: The LSTM autoencoder learns typical patterns in these sequences.
- **Anomaly Detection**: High reconstruction errors highlight frames with unusual facial expressions, indicating potential anomalies.

This approach effectively captures temporal dependencies and subtle changes in facial expressions, providing robust anomaly detection.

### Methods of Anomaly Detection:
1. **Using All Features**: Considers feature components and emotion scores as input features.
2. **Using Reduced Components**: Uses UMAP to reduce the dimensionality of facial embeddings before inputting them into the LSTM autoencoder.
3. **Using Full-Dimensional Embeddings**: Directly uses the raw facial embeddings without dimensionality reduction.

Each method provides a different perspective on the data, enhancing our capability to detect subtle and varied anomalies in facial expressions.

## An Example from a Death Sentence Verdict
Wade Wilson, a 30-year-old from Fort Myers, Florida, was convicted in June 2024 for the October 2019 murders of Kristine Melton and Diane Ruiz in Cape Coral. During the trial, Wilson exhibited a "smug, soulless" demeanor, displaying a lack of emotion throughout the proceedings. The jury recommended the death penalty, with the final sentencing set for July 23, 2024. Despite Wilson's cold demeanor, the LSTM autoencoder detected emotional anomalies during critical moments, suggesting that mentions of the death penalty had a marked impact on Wilson.

Sources:
1. [Fox News](https://www.foxnews.com/us/florida-double-murderer-viral-smug-soulless-courtroom-demeanor)
2. [WINK News](https://winknews.com/2024/06/13/wade-wilsons-lack-emotion-double-murder-trial/)
3. [YouTube](https://www.youtube.com/watch?v=8j8psgKXmRg)

## Setup Parameters
- **DESIRED_FPS**: Frames per second to analyze (lower for faster processing).
- **batch_size**: Batch size for processing.

## Output
- **Organized Faces**: Faces organized by detected persons in the `organized_faces` folder.
- **Anomalies Detection Results**: Results as a CSV file in the project directory.

## Face Extraction and Alignment
The algorithm extracts faces from the video, aligns, and normalizes them using MediaPipe and MTCNN.

## Feature Embedding Extraction
Utilizes the InceptionResnetV1 model to extract facial features and FER to detect emotions.

## Clustering and Outlier Detection
Clusters faces to organize them by person.

## LSTM Autoencoder for Anomaly Detection
Trains an LSTM autoencoder to identify anomalies in facial expressions over time, helping capture temporal dependencies and irregularities in the sequence of facial expressions and feature embeddings.

## Conclusion
This tool offers robust solutions for detecting emotional anomalies in video-based facial expressions, beneficial for both forensic analysis and HUMINT operations. By leveraging advanced computer vision techniques and the power of LSTM autoencoders, it provides timely and crucial insights into human behavior.
