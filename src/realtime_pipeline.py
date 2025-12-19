import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ACTION_CLASSES = ["Walking", "Waving", "Standing", "Sitting"]
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

SEQ_LENGTH = 16
ACTION_IMG_SIZE = 128
EMOTION_IMG_SIZE = 48

ACTION_PRED_INTERVAL = 5


def _load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def _draw_text(img, text: str, org: tuple[int, int], font_scale: float = 0.7, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, 2, cv2.LINE_AA)


def _preprocess_emotion_face(gray_frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = face_bbox
    roi = gray_frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, (EMOTION_IMG_SIZE, EMOTION_IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    roi = roi.reshape(1, EMOTION_IMG_SIZE, EMOTION_IMG_SIZE, 1)
    return roi


def _preprocess_action_frame(bgr_frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(bgr_frame, (ACTION_IMG_SIZE, ACTION_IMG_SIZE), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame


def _select_largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    return int(x), int(y), int(w), int(h)


def main():
    tf.get_logger().setLevel("ERROR")

    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"

    action_model_path = models_dir / "action_model_adam.keras"
    emotion_model_path = models_dir / "emotion_model_best.keras"

    action_model = _load_model(action_model_path)
    emotion_model = _load_model(emotion_model_path)

    face_cascade_path = base_dir / "models" / "haarcascade_frontalface_default.xml"
    if not face_cascade_path.exists():
        raise FileNotFoundError(f"HaarCascade not found: {face_cascade_path}")

    face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load HaarCascade: {face_cascade_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buffer = deque(maxlen=SEQ_LENGTH)
    frame_count = 0
    last_action_pred_frame = -ACTION_PRED_INTERVAL

    action_label = "Collecting..."
    action_conf = 0.0

    emotion_label = "No face"
    emotion_conf = 0.0

    last_time = time.time()
    fps_ema = 0.0

    window_name = "CSE480 Real-Time Pipeline"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_count += 1

        now = time.time()
        dt = now - last_time
        last_time = now
        fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * fps)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        best_face = _select_largest_face(faces)

        if best_face is not None:
            x, y, w, h = best_face
            try:
                face_input = _preprocess_emotion_face(gray, best_face)
                e_probs = emotion_model.predict(face_input, verbose=0)[0]
                e_idx = int(np.argmax(e_probs))
                emotion_label = EMOTION_CLASSES[e_idx]
                emotion_conf = float(e_probs[e_idx])
            except Exception:
                emotion_label = "Face err"
                emotion_conf = 0.0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            _draw_text(frame, f"{emotion_label} {emotion_conf * 100:.1f}%", (x, label_y), font_scale=0.6, color=(0, 255, 0))
        else:
            emotion_label = "No face"
            emotion_conf = 0.0

        buffer.append(_preprocess_action_frame(frame))

        if len(buffer) == SEQ_LENGTH and (frame_count - last_action_pred_frame) >= ACTION_PRED_INTERVAL:
            seq = np.stack(buffer, axis=0).astype(np.float32)
            seq = np.expand_dims(seq, axis=0)
            a_probs = action_model.predict(seq, verbose=0)[0]
            a_idx = int(np.argmax(a_probs))
            action_label = ACTION_CLASSES[a_idx]
            action_conf = float(a_probs[a_idx])
            last_action_pred_frame = frame_count

        _draw_text(frame, f"Status: {action_label} ({action_conf * 100:.0f}%)", (10, 30), font_scale=0.8, color=(255, 255, 255))

        fps_text = f"FPS: {fps_ema:.1f}"
        x_fps = max(10, frame.shape[1] - 170)
        _draw_text(frame, fps_text, (x_fps, 30), font_scale=0.7, color=(255, 255, 0))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
