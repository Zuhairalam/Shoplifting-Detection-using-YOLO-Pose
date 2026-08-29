# Shoplifting Detection using YOLO Pose

Computer vision-based shoplifting detection using YOLO pose estimation and a neural network classifier.

## Project Overview

This project detects shoplifting activity from video footage using human body pose information.

YOLO is used for person detection and pose estimation. The extracted human body keypoints are then used as features for a neural network classifier to classify the activity as **Shoplifting** or **Non-Shoplifting**.

## Project Pipeline

Video  
↓  
Frame Extraction  
↓  
Person Detection  
↓  
Person Cropping  
↓  
Pose Estimation  
↓  
Keypoint Extraction  
↓  
Neural Network Classification  
↓  
Shoplifting / Non-Shoplifting Prediction

## Methodology

### 1. Frame Extraction

The input videos are converted into individual frames using OpenCV.

### 2. Person Detection

YOLO is used to detect people in each frame. The detected person is then cropped from the frame.

### 3. Pose Estimation

YOLO Pose is used to extract human body keypoints from the detected person.

The model provides **17 body keypoints**, with each keypoint represented using its `(x, y)` coordinates.

Therefore:

**17 keypoints × 2 coordinates = 34 features**

These 34 features are used as the input to the classification model.

### 4. Classification

A neural network implemented using PyTorch is trained using the extracted pose features.

The model performs binary classification:

- `0` → Non-Shoplifting
- `1` → Shoplifting

### 5. Video Prediction

The trained classifier can be applied to video frames to generate shoplifting/non-shoplifting predictions.

## Dataset

This project uses the [Shoplifting Videos Dataset](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset) available on Kaggle.

The dataset contains videos belonging to:

- Shoplifters
- Non-Shoplifters

### Dataset Preparation

The original videos contain different types of activities, including casual walking, touching or examining objects, and shoplifting-related actions.

For this project, the relevant portions of the videos were **manually clipped** before training.

The prepared videos were organized into two categories:

- **Shoplifting** – clips containing shoplifting activity
- **Non-Shoplifting** – clips containing normal, non-shoplifting activity

The prepared video clips were then converted into frames and processed through the detection, pose estimation, and classification pipeline.

> The dataset is not included in this repository. Please download it from Kaggle before running the notebook.

## Technologies Used

- Python
- OpenCV
- YOLO
- YOLO Pose
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

## Model

The classification model is a feed-forward neural network implemented using PyTorch.

**Input:** 34 pose features

**Output:** Binary classification

- Non-Shoplifting
- Shoplifting

## Results

The current experiment achieved approximately **95.45% test accuracy**
on the held-out test samples.

However, the current evaluation uses a random frame-level train/test
split. Since frames from the same video can be highly similar, this
result may not fully represent performance on completely unseen videos.

Further evaluation using video-level train/test separation is required
to measure real-world generalization.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Zuahiralalam/Shoplifting-Detection-using-YOLO-Pose.git
cd Shoplifting-Detection-using-YOLO-Pose
