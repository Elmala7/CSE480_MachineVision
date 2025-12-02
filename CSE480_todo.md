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
- [ ] **Data Prep (Emotion):** Process FER-2013 (Resize to 48x48).
- [ ] **Architecture Experiment:**
    - [ ] Train a simple VGG-style block.
    - [ ] Train a ResNet-style block.
    - [ ] Compare accuracy on FER-2013 test set.
- [ ] **Selection:** Save the best Emotion model (`emotion_best.keras`).

## Phase 4: Integration & Real-Time (Milestone 2 Core)
- [ ] **Pipeline:**
    - [ ] Capture Webcam -> Detect Face -> Crop -> Predict Emotion.
    - [ ] Buffer Frames -> Predict Action.
- [ ] **Overlay:** Draw labels and FPS on screen.
- [ ] **Optimization:** Thread the camera capture to improve FPS on M1.

## Phase 5: Final Evaluation & Report
- [ ] **Test Set Eval:** Run final accuracy check on the *unseen* test sets for both models.
- [ ] **Discussion:** Write the "Justification" section explaining *why* Adam/SGD worked better.
- [ ] **Preprocessing Page:** Write the dedicated 1-page section on data cleaning.