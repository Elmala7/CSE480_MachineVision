# CSE480 Machine Vision Project

**Action & Emotion Recognition using Deep Learning**

A real-time system for recognizing human actions and facial emotions using hybrid CNN-LSTM and CNN architectures, optimized for Mac M1.

## Project Overview

This project implements a dual-branch recognition system:

- **Emotion Recognition Model**: CNN-based architecture (VGG, ResNet, or EfficientNet) for facial emotion classification using the FER-2013 dataset
  - Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
  - Input: 48x48 grayscale face crops

- **Action Recognition Model**: Hybrid CNN-LSTM architecture for temporal action recognition
  - Spatial Feature Extractor: ResNet or MobileNet for frame-level features
  - Temporal Sequence Learner: LSTM layers for motion analysis over time
  - Classes: Walking, Standing, Waving, Sitting
  - Input: Sequences of 10-20 frames

## Technical Stack

- **Language**: Python 3.x
- **Deep Learning**: TensorFlow/Keras (optimized for Mac M1)
- **Computer Vision**: OpenCV (`cv2`)
- **Libraries**: NumPy, Pandas, Matplotlib
- **Environment**: `mecha_env`

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CSE480_MachineVision.git
cd CSE480_MachineVision
```

### 2. Create Virtual Environment

```bash
conda create -n mecha_env python=3.10
conda activate mecha_env
```

### 3. Install Dependencies

```bash
pip install opencv-python tensorflow keras numpy pandas matplotlib
```

For Mac M1 optimization:
```bash
pip install tensorflow-metal
```

### 4. Initialize Project Structure

```bash
python initialize_project.py
```

### 5. Download Datasets

- **FER-2013**: Download from Kaggle and extract to `data/raw/fer2013/`
- **UCF-101**: Download and extract action clips to `data/raw/ucf101/`

## Project Structure

```
CSE480_MachineVision/
├── data/
│   ├── raw/           # Original datasets (FER-2013, UCF-101)
│   └── processed/     # Preprocessed images and sequences
├── src/               # Source code
│   ├── preprocessing.py
│   └── real_time_inference.py (to be implemented)
├── models/            # Saved model files (.keras, .h5)
├── notebooks/         # Jupyter notebooks for experiments
├── reports/           # Milestone reports
├── initialize_project.py
├── CSE480_specifications.md
└── CSE480_todo.md
```

## Milestones

### Milestone 1: Models & Training (Week 11)
- Data preparation and preprocessing
- Train Emotion CNN and Action CNN-LSTM models
- **Optimizer Comparison**: Compare SGD, Adam, and Adagrad optimizers
- Generate training results and reports

### Milestone 2: Real-Time Integration (Week 14)
- Real-time webcam pipeline
- Face detection and emotion recognition
- Action recognition on frame sequences
- FPS and latency evaluation
- Performance optimization

## Usage

### Preprocessing

```python
from src.preprocessing import process_face

# Process a face image for emotion recognition
processed_face = process_face("path/to/image.jpg")
# Returns: numpy array of shape (48, 48)
```

### Training

For the Action Recognition (Milestone 1) pipeline:

```bash
python src/make_dataset_action.py   # Prepare action dataset (UCF + custom)
python src/train_action_model.py    # Train CNN-LSTM with SGD, Adam, Adagrad
python src/check_models.py          # Verify saved models and run test inference
```

The optimizer comparison plot is saved to `reports/milestone1_optimizer_comparison.png`.

### Real-Time Inference (To be implemented)

```bash
python src/real_time_inference.py
```

## Development

For detailed specifications and roadmap, see:
- `CSE480_specifications.md` - Technical requirements and architecture details
- `CSE480_todo.md` - Development roadmap and task list

## Notes

- Large dataset files are excluded from git (see `.gitignore`)
- Trained model files should be stored separately or uploaded to GitHub Releases
- For dataset downloads, refer to the respective dataset documentation

## License

[Add your license here]

## Author

[Your name/team]

