# Project Roadmap (CSE480)

## Phase 1: Milestone 1 Setup (Action Only)
- [x] **Data Prep (Action):**
    - [x] Filter UCF-101 to 4 classes (Walking, Waving, Standing, Sitting).
    - [x] **Augmentation:** Add logic to flip/rotate frames during processing.
    - [x] Split into `train` and `test` (held-out) sets.
- [x] **Model Definition:** Create `build_action_model(optimizer)` function (CNN + LSTM 1-2 layers).

## Phase 2: The Optimizer Experiment (Milestone 1 Core)
- [x] **Train Loop:**
    - [x] Train Action Model with **SGD**. Save history & model.
    - [x] Train Action Model with **Adam**. Save history & model.
    - [x] Train Action Model with **Adagrad**. Save history & model.
- [x] **Analysis:**
    - [x] Plot all 3 loss curves on one graph.
    - [x] Select best model for the real-time phase (Selected Adam due to steepest loss convergence).
- [x] **Report M1:** Generate charts and write the "Methods" section.

## Phase 3: Milestone 2 Setup (Emotion)
- [x] **Data Prep (Emotion):** Process FER-2013 (Resize to 48x48).
- [x] **Architecture Experiment:**
    - [x] Train a simple VGG-style block (see `src/train_emotion_model.py` → `build_vgg_model()`).
    - [x] Train a ResNet-style block (see `src/train_emotion_model.py` → `build_resnet_model()`).
    - [x] Compare accuracy on FER-2013 test set. Best model saved based on validation accuracy.
- [x] **Selection:** Save the best Emotion model (`emotion_model_best.keras`). Architecture comparison plot saved to `reports/milestone2_architecture_comparison.png`.

## Phase 4: Integration & Real-Time (Milestone 2 Core)
- [x] **Pipeline:**
    - [x] Capture Webcam -> Detect Face (HaarCascade) -> Crop -> Predict Emotion (see `src/realtime_pipeline.py`).
    - [x] Buffer Frames -> Predict Action using CNN-LSTM (every 5 frames for performance).
- [x] **Overlay:** Draw labels, confidence scores, and FPS on screen.
- [x] **Optimization:** Frame buffering and throttled action predictions to improve FPS