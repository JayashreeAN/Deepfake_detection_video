# Deepfake Detection System

A deep learning-based system to detect and classify deepfake videos with high accuracy using Xception neural networks.

## 📊 Features

- **Real-time Video Analysis**: Processes video frames and detects deepfakes
- **Face Detection**: Automatically detects and extracts faces from videos
- **Confidence Scoring**: Provides confidence percentage for each prediction
- **Multiple Models**: Supports various pre-trained Xception models
- **Frame-by-frame Analysis**: Analyzes multiple frames for robust detection

## 🎯 Model Performance

- **Accuracy**: 89.25%
- **Confidence Score Range**: 0.0 - 1.0
- **Supported Video Formats**: MP4, AVI, MOV, MKV

## 📋 Score Interpretation

0.0 - 0.35 = DEFINITELY REAL ✅
0.35 - 0.50 = PROBABLY REAL ✅
0.50 - 0.65 = PROBABLY FAKE ⚠️
0.65 - 1.0 = DEFINITELY FAKE ❌

## Quick Setup (First Time Only)
Step 1: Clone Repository
Step 2: Open in VS Code
Step 3: Create Virtual Environment
Step 4: Activate Virtual Environment
Step 5: Install Dependencies/requirements.txt
step 6: run this below command to upload the video path to get the result whether the uplaoded video is real or fake 
## python upload_and_test.py 