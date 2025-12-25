# CSE480 Machine Vision Project

**Action & Emotion Recognition using Deep Learning** ✅

A real-time system for recognizing human actions and facial emotions using CNN and CNN–LSTM architectures. This repo contains preprocessing, dataset builders, training scripts, and a working real-time demo that runs on a webcam.

---

## Project Overview

This project implements a dual-branch recognition system:

- **Emotion Recognition (FER)**
  - Architectures implemented: Simple VGG-like CNN, Mini-ResNet (training in `src/train_emotion_model.py`).
  - Classes: angry, disgust, fear, happy, sad, surprise, neutral
  - Input: 48×48 grayscale face crops (see `src/preprocessing.py` → `process_face`)

- **Action Recognition**
  - Architecture implemented: MobileNetV2 backbone + LSTM temporal aggregator (see `src/train_action_model.py`).
  - Classes (default): Walking, Waving, Standing, Sitting
  - Input: sequences of frames (default SEQ_LENGTH = 16, frame size 128×128)

---

## Technical Stack 🔧

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow / Keras (TensorFlow 2.x)
- **Computer Vision**: OpenCV (`cv2`)
- **Libraries**: NumPy, Pandas, Matplotlib
- **Environment name (suggested)**: `mecha_env`

---

## Quick Setup

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CSE480_MachineVision.git
cd CSE480_MachineVision
```

2. Create & activate a conda environment

```bash
conda create -n mecha_env python=3.10
conda activate mecha_env
```

3. Install dependencies

```bash
pip install opencv-python tensorflow numpy pandas matplotlib
```

4. Initialize project skeleton (creates `data/processed/` etc.)

```bash
python initialize_project.py
```

5. Download the datasets
- FER-2013 (Kaggle) → extract to `data/raw/fer2013/`
- UCF-101 or your action clips → place under `data/raw/ucf101/` or `data/raw/custom/`

---

## Project Structure (key files)

```
CSE480_MachineVision/
├── data/
│   ├── raw/           # Original datasets (FER-2013, UCF-101, custom)
│   └── processed/     # Preprocessed arrays (.npy) used by training scripts
├── src/
│   ├── preprocessing.py      # process_face(...) utility
│   ├── make_dataset_emotion.py
│   ├── make_dataset_action.py
│   ├── train_emotion_model.py # trains VGG and mini-ResNet; saves best weights
│   ├── train_action_model.py  # trains MobileNetV2+LSTM with multiple optimizers
│   ├── realtime_pipeline.py   # real-time webcam demo (runs inference)
│   ├── check_models.py        # quick sanity-check / inference helper
│   └── inspect_data.py
├── models/            # saved models (.keras) and weights (.h5), HaarCascade xml
├── reports/           # plots and result images (e.g. optimizer/architecture comparisons)
├── notebooks/
├── initialize_project.py
└── Docs/
```

---

## Usage Examples

### Preprocessing

```python
from src.preprocessing import process_face
img48 = process_face("path/to/image.jpg")
# returns a grayscale numpy array of shape (48, 48)
```

### Prepare datasets

```bash
python src/make_dataset_emotion.py   # saves to data/processed/emotion_*.npy
python src/make_dataset_action.py    # saves to data/processed/action_*.npy
```

### Training

Emotion models (VGG and Mini-ResNet):

```bash
python src/train_emotion_model.py
# saves best weights as: models/emotion_vgg_best.weights.h5, models/emotion_mini_resnet_best.weights.h5
# and final best model: models/emotion_model_best.keras
```

Action model (Optimizer comparison: SGD, Adam, Adagrad):

```bash
python src/train_action_model.py
# saves models: models/action_model_sgd.keras, models/action_model_adam.keras, models/action_model_adagrad.keras
# optimizer comparison plot: reports/milestone1_optimizer_comparison.png
```

### Real-time Demo (webcam)

```bash
python src/realtime_pipeline.py
```

- Requirements: `models/emotion_model_best.keras` and one of `models/action_model_*.keras` present, and `models/haarcascade_frontalface_default.xml`.
- Controls: Press `q` or `Esc` to quit.
- Notes: Default sequence length is 16 frames and frame size is 128×128; action predictions are made every few frames to reduce latency.

---

## Outputs & Reports

- `reports/milestone1_optimizer_comparison.png` — action optimizer comparison
- `reports/milestone2_architecture_comparison.png` — emotion architecture comparison
- Saved model files are kept in `models/`

---