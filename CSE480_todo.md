# Project Roadmap (CSE480)

## Phase 1: Data Setup (Milestone 1)
- [ ] **Download:** Get FER-2013 and UCF-101.
- [ ] **Filter UCF-101:** Extract only `WalkingWithDog` (Walking), `StandUp` (Standing), `HandWaving` (Waving), and `SoccerPenalty` or similar (Sitting).
- [ ] **Preprocessing Script:**
    -   Images: Resize to 48x48.
    -   Videos: Extract 16 frames per clip, resize to 128x128.
- [ ] **Data Check:** Verify shapes `(N, 48, 48, 1)` for emotion and `(N, 16, 128, 128, 3)` for action.

## Phase 2: Model Training & Experimentation (The Experiment)
- [ ] **Emotion Model:** Define a CNN (e.g., Mini-VGG).
- [ ] **Action Model:** Define CNN-LSTM (MobileNet + LSTM).
- [ ] **Optimizer Loop:**
    -   Train Emotion Model with **SGD**. Save history.
    -   Train Emotion Model with **Adam**. Save history.
    -   Train Emotion Model with **Adagrad**. Save history.
- [ ] **Compare:** Plot the 3 Loss Curves on one graph using Matplotlib.
- [ ] **Select Best:** Save the best performing models as `emotion_best.keras` and `action_best.keras`.

## Phase 3: Real-Time Application (Milestone 2)
- [ ] **Webcam Script:** Setup `cv2.VideoCapture(0)`.
- [ ] **Face Detection:** Implement Haar Cascade or MediaPipe.
- [ ] **Inference Logic:**
    -   Pass face crop to Emotion Model.
    -   Pass frame buffer (list of 16 images) to Action Model.
- [ ] **Overlay:** Draw `Action: Walking` and `Emotion: Happy` on screen.
- [ ] **FPS Counter:** Add `cv2.putText` showing `1 / (time_now - time_prev)`.

## Phase 4: Reporting
- [ ] **Report M1:** Write 1 page on Data, 2-3 pages on Algorithms, and include the **Optimizer Comparison Plots**.
- [ ] **Report M2:** Write about the Real-Time Latency and FPS on your Mac M1.