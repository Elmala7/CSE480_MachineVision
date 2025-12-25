# Project Specifications: Action & Emotion Recognition System

## [cite_start]1. Milestone 1: Action Recognition (Week 11) [cite: 14-31]
**Objective:** Build and optimize the CNN-LSTM model for human activity recognition.

### A. Data Preparation
* **Dataset:** UCF-101 (or custom/subset).
* **Classes:** Walking, Standing, Waving, Sitting.
* **Preprocessing:**
    * Extract 10-20 frames per clip.
    * Resize/Normalize.
    * [cite_start]**Augmentation (Required):** Apply random flips, rotations, and brightness adjustments[cite: 20].

### B. Model Architecture
* [cite_start]**Type:** Hybrid **CNN-LSTM**[cite: 11].
* **Spatial:** ResNet or MobileNet (Feature Extractor).
* **Temporal:** LSTM (1-2 layers).

### [cite_start]C. The Optimizer Experiment (Action Model Only) [cite: 27-31]
You must train the Action Model 3 separate times to compare:
1.  **SGD**
2.  **Adam**
3.  **Adagrad**
* **Metrics:** Plot accuracy, training time, and loss evolution for each.

---

## [cite_start]2. Milestone 2: Emotion & Real-Time Integration (Week 14) [cite: 38-49]
**Objective:** Add the Emotion model and build the real-time pipeline.

### A. Emotion Model (New for M2)
* **Dataset:** FER-2013.
* [cite_start]**Architecture Comparison:** Compare 2-3 architectures (e.g., VGG vs. ResNet vs. EfficientNet)[cite: 41, 49].

### B. Real-Time Pipeline
* **Integration:** Combine Action (CNN-LSTM) and Emotion (CNN) into one loop.
* **Input:** Live Webcam using OpenCV.
* **Output:** Overlay Action/Emotion labels + FPS Counter.

### C. Evaluation
* [cite_start]**Test Set:** Evaluate both models on **unseen test clips** (held-out data)[cite: 47].
* **Performance:** Report FPS and Latency.

---

## [cite_start]3. Deliverables Structure [cite: 54-60]
* **Report 1 (Action Focus):**
    * Problem Definition (1 Page).
    * Data Cleaning & Preprocessing (1 Page).
    * Methods (2-3 Pages).
    * **Result:** Optimizer Comparison Plots.
* **Report 2 (Full System):**
    * Emotion Model Design & Architecture Comparison.
    * Real-Time Integration & Logic.
    * **Discussion:** Justify performance