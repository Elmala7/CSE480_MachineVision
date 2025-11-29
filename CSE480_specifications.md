# Project Specifications: Action & Emotion Recognition System

## 1. Technical Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow/Keras (Optimized for M1).
* **Vision:** OpenCV (`cv2`) for real-time capture.
* **Libraries:** NumPy, Pandas, Matplotlib.

## 2. Milestone 1: Models & Training (Week 11)
### A. Data Preparation
* **Action Data:** UCF-101 (or custom).
    * **Required Classes:** Walking, Standing, Waving, **Sitting**.
    * **Preprocessing:** Extract short sequences (10-20 frames per clip).
* **Emotion Data:** FER-2013.
    * **Preprocessing:** Resize/Normalize 48x48 grayscale images.

### B. Model Architectures
* **Action Recognition:** Hybrid **CNN-LSTM**.
    * **Spatial:** CNN (ResNet or MobileNet) to extract features from frames.
    * **Temporal:** LSTM (1-2 layers) to analyze motion over time.
* **Emotion Recognition:** Standard **CNN** (VGG, ResNet, or EfficientNet).

### C. Optimization Experiment (Crucial)
You **MUST** train the models using these 3 distinct optimizers and compare them:
1.  **SGD** (Stochastic Gradient Descent).
2.  **Adam**.
3.  **Adagrad**.
* **Metrics:** Record accuracy, training time, and loss evolution for *each* optimizer.

## 3. Milestone 2: Real-Time Integration (Week 14)
* **Pipeline:**
    1.  Capture Webcam Feed.
    2.  **Face Branch:** Detect Face -> Crop -> Predict Emotion.
    3.  **Action Branch:** Buffer last 16 frames -> Predict Action.
    4.  **UI:** Overlay labels on the live video.
* **Evaluation:** Measure **FPS** (Frames Per Second) and Latency.

## 4. Deliverables
* **Report 1 (Milestone 1):** Problem definition, Data cleaning, Methods, Training Results (Optimizer comparison).
* **Report 2 (Milestone 2):** Real-time architecture, FPS performance, Challenges.
* **Code:** Clean repository with `src/`, `models/`, and `notebooks/`.