<div align="center">
  <h1>CSE480: Machine Vision System</h1>
  <p><strong>Action & Emotion Recognition using Deep Learning</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
    <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  </p>
</div>

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Results & Evaluation](#-results--evaluation)

---

## 🚀 About the Project

This repository contains the implementation of a robust, real-time dual-branch computer vision system designed to simultaneously recognize human actions and facial emotions. Leveraging modern Deep Learning architectures (CNNs and CNN-LSTMs), the project provides a complete pipeline from dataset preprocessing and model training to real-time inference via webcam.

## ✨ Key Features

- **Dual-Branch Pipeline**: Concurrently processes video streams for temporal action recognition and spatial facial emotion detection.
- **Real-Time Inference**: Optimized webcam pipeline for low-latency live predictions.
- **Comprehensive Training Suite**: Includes scripts for dataset creation, model training, and performance evaluation.
- **Modular Design**: Clean, well-structured codebase making it easy to swap architectures or integrate new datasets.

---

## 🧠 Model Architecture

### Emotion Recognition (FER)
Focuses on spatial feature extraction from facial crops.
- **Architectures**: VGG-inspired CNN, Mini-ResNet.
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Input Pipeline**: Extracts and processes 48×48 grayscale facial regions.

### Action Recognition
Captures temporal dynamics and movement patterns across video sequences.
- **Architecture**: MobileNetV2 (Spatial Feature Extractor) + LSTM (Temporal Aggregator).
- **Classes**: Walking, Waving, Standing, Sitting (Customizable).
- **Input Pipeline**: Processes sequential frames (Default: 16 frames, 128×128 resolution).

---

## 🛠️ Getting Started

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) or Python 3.10+ installed on your system. 

### 1. Installation

Clone the repository and set up the environment:

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/CSE480_MachineVision.git
cd CSE480_MachineVision

# Create and activate a conda environment
conda create -n mecha_env python=3.10
conda activate mecha_env

# Install dependencies
pip install opencv-python tensorflow numpy pandas matplotlib
```

### 2. Initialization

Set up the required directory structure for data, models, and reports:

```bash
python initialize_project.py
```

### 3. Dataset Preparation

Download the required datasets and place them in the corresponding `data/raw/` subdirectories:
- **FER-2013** (Kaggle): Extract contents into `data/raw/fer2013/`
- **UCF-101** (or custom action clips): Place under `data/raw/ucf101/` or `data/raw/custom/`

Convert the raw datasets into optimized NumPy arrays for training:

```bash
python src/make_dataset_emotion.py
python src/make_dataset_action.py
```

---

## 💻 Usage Guide

### Model Training

**Train Emotion Models (VGG & Mini-ResNet):**
```bash
python src/train_emotion_model.py
```
*Outputs are saved to `models/` (e.g., `emotion_model_best.keras`, `emotion_vgg_best.weights.h5`).*

**Train Action Models:**
Compares SGD, Adam, and Adagrad optimizers.
```bash
python src/train_action_model.py
```
*Outputs are saved to `models/` along with performance plots in `reports/`.*

### Real-Time Webcam Demo

Run the live inference pipeline:
```bash
python src/realtime_pipeline.py
```
*Note: Requires trained models (`emotion_model_best.keras` and an action model) in the `models/` directory, along with the HaarCascade XML file. Press `q` or `Esc` to exit the stream.*

---

## 📂 Project Structure

```text
CSE480_MachineVision/
├── data/
│   ├── raw/                 # Raw datasets (FER-2013, UCF-101)
│   └── processed/           # Processed datasets (.npy format)
├── src/
│   ├── preprocessing.py     # Image/Video processing utilities
│   ├── make_dataset_*.py    # Dataset creation scripts
│   ├── train_*_model.py     # Model training scripts
│   ├── realtime_pipeline.py # Live webcam inference
│   └── check_models.py      # Inference sanity checks
├── models/                  # Saved models (.keras, .h5) & Cascade XMLs
├── reports/                 # Training plots and evaluation metrics
├── notebooks/               # Jupyter notebooks for exploration
├── Docs/                    # Additional documentation
└── initialize_project.py    # Environment setup script
```

---

## 📊 Results & Evaluation

Training progress and model comparisons are automatically generated and saved during the training phase. Check the `reports/` directory for visual insights:
- `milestone1_optimizer_comparison.png`: Action model optimizer performance.
- `milestone2_architecture_comparison.png`: Emotion model architecture comparison.