# Facial-Expression-Anomality-Detection

## Introduction
This repository contains an algorithm for detecting anomalies in facial expressions over the timeline of a video using time series analysis. The tool extracts faces from video frames at the desired frames per second (FPS), detects facial features, and analyzes facial expressions to identify anomalies. This is particularly useful for forensic analysis and human intelligence (HUMINT) operations.

## Key Features
- **Face Extraction**: Extracts faces from video frames at a defined FPS.
- **Face Alignment**: Aligns and normalizes faces for consistent analysis.
- **Feature Embeddings**: Extracts facial feature embeddings using the VGG-Face model.
- **Emotion Detection**: Identifies facial expressions and categorizes emotions.
- **Anomaly Detection**: Uses various techniques and models to detect anomalies in facial expressions.

## Practical Applications
### Forensic Analysis
- Identify suspicious behavior in surveillance footage.
- Detect stress or duress in interrogation videos.

### Human Intelligence (HUMINT)
- Analyze micro-expressions for lie detection.
- Monitor and assess emotional states in communications.

## Theoretical Background

### VGG-Face Model
The VGG-Face model is a deep convolutional neural network trained on the VGGFace2 dataset. It is widely used for facial recognition and attribute prediction. 

### Micro-Expressions and Paul Ekman’s Theory
Paul Ekman’s work on facial expressions of emotion identified universal micro-expressions that reveal true emotions. Detecting these fleeting expressions is crucial in understanding underlying emotions.

## Dependencies
- `torch`
- `facenet-pytorch`
- `mediapipe`
- `FER`
- `sklearn`
- `umap-learn`
- `tqdm`
- `opencv-python`
- `scipy`
- `pandas`

## Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/username/repo.git
cd repo
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Run the Script
Set user-defined parameters in the script and execute:
```python
python detect_anomalies.py
```

### Output
- Extracted faces saved in the `aligned_faces` folder.
- Organized faces by detected persons in the `organized_faces` folder.
- Anomalies detection results as a CSV file in the project directory.

## Code Summary
### Face Extraction and Alignment
The algorithm extracts faces from the video, aligns, and normalizes them using MediaPipe and MTCNN.

### Feature Embedding Extraction
Utilizes the VGG-Face model to extract facial features and FER to detect emotions.

### Clustering and Outlier Detection
Clusters faces to organize them by person and uses models like Isolation Forest, One-Class SVM, and Local Outlier Factor to detect anomalies.

### Grid Search and Anomaly Detection
Optimizes parameters using grid search and identifies top anomalies based on clustering and emotion data over time.

## Example Output
Here is an example of an anomaly detection output:
```
Time of top 10 anomalies:
- 00:01:23
- 00:02:45
- 00:05:10

Details saved to 'anomaly_detection_results_person_1.csv'
```

## Conclusion
This tool offers robust solutions for detecting emotional anomalies in video-based facial expressions, beneficial for both forensic analysis and HUMINT operations. By leveraging advanced computer vision techniques, it provides timely and crucial insights into human behavior.
