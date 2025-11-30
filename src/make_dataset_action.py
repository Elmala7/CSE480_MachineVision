import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_SIZE = 128
SEQ_LENGTH = 16
CLASSES = ["Walking", "Waving", "Standing", "Sitting"]
MIN_CUSTOM_SAMPLES = 50
CUSTOM_STEP = 60


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
    cap.release()
    return frames


def sample_sequence_from_video(frames):
    n = len(frames)
    if n == 0:
        return None
    if n >= SEQ_LENGTH:
        indices = np.linspace(0, n - 1, SEQ_LENGTH).astype(int)
    else:
        indices = [i % n for i in range(SEQ_LENGTH)]
    sequence = [frames[i] for i in indices]
    return np.stack(sequence, axis=0)


def slice_long_video_into_sequences(frames):
    sequences = []
    n = len(frames)
    if n == 0:
        return sequences
    if n < SEQ_LENGTH:
        seq = sample_sequence_from_video(frames)
        if seq is not None:
            sequences.append(seq)
    else:
        for start in range(0, n - SEQ_LENGTH + 1, CUSTOM_STEP):
            window = frames[start:start + SEQ_LENGTH]
            if len(window) == SEQ_LENGTH:
                sequences.append(np.stack(window, axis=0))
    if len(sequences) == 0:
        return sequences
    while len(sequences) < MIN_CUSTOM_SAMPLES:
        for seq in list(sequences):
            sequences.append(seq.copy())
            if len(sequences) >= MIN_CUSTOM_SAMPLES:
                break
    return sequences


def augment_horizontal_flip(sequence):
    return np.flip(sequence, axis=2)


def load_ucf101_sequences(ucf_root, class_to_idx):
    folder_to_label = {
        "WalkingWithDog": "Walking",
    }
    video_label_pairs = []
    for folder_name, label in folder_to_label.items():
        folder_path = ucf_root / folder_name
        if not folder_path.is_dir():
            continue
        for pattern in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
            for video_path in folder_path.glob(pattern):
                video_label_pairs.append((video_path, label))
    sequences = []
    labels = []
    for video_path, label in tqdm(video_label_pairs, desc="Processing UCF101 videos"):
        frames = load_video_frames(video_path)
        seq = sample_sequence_from_video(frames)
        if seq is None:
            continue
        sequences.append(seq)
        labels.append(class_to_idx[label])
        flipped = augment_horizontal_flip(seq)
        sequences.append(flipped)
        labels.append(class_to_idx[label])
    return sequences, labels


def load_custom_sequences(custom_root, class_to_idx):
    sequences = []
    labels = []

    # Handle Waving from custom (can be a folder of clips or a single long video)
    waving_dir_candidates = [
        custom_root / "HandWaving",
        custom_root / "Waving",
    ]
    waving_file_candidates = [
        custom_root / "HandWaving.mov",
        custom_root / "HandWaving.mp4",
        custom_root / "Waving.mov",
        custom_root / "Waving.mp4",
    ]
    waving_path = None
    for candidate in waving_dir_candidates + waving_file_candidates:
        if candidate.exists():
            waving_path = candidate
            break

    if waving_path is not None:
        if waving_path.is_dir():
            video_paths = []
            for pattern in ("*.avi", "*.mp4", "*.mov", "*.mkv"):
                for video_path in waving_path.glob(pattern):
                    video_paths.append(video_path)
            for video_path in tqdm(video_paths, desc="Processing custom Waving (folder)"):
                frames = load_video_frames(video_path)
                seq = sample_sequence_from_video(frames)
                if seq is None:
                    continue
                sequences.append(seq)
                labels.append(class_to_idx["Waving"])
                flipped = augment_horizontal_flip(seq)
                sequences.append(flipped)
                labels.append(class_to_idx["Waving"])
        else:
            frames = load_video_frames(waving_path)
            seqs = slice_long_video_into_sequences(frames)
            for seq in tqdm(seqs, desc="Processing custom Waving (file)", leave=False):
                sequences.append(seq)
                labels.append(class_to_idx["Waving"])
                flipped = augment_horizontal_flip(seq)
                sequences.append(flipped)
                labels.append(class_to_idx["Waving"])

    label_to_files = {
        "Standing": ["Standing.mov", "Standing.mp4"],
        "Sitting": ["Sitting.mov", "Sitting.mp4"],
    }
    for label, filenames in label_to_files.items():
        video_path = None
        for name in filenames:
            candidate = custom_root / name
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            continue
        frames = load_video_frames(video_path)
        seqs = slice_long_video_into_sequences(frames)
        for seq in tqdm(seqs, desc=f"Processing custom {label}", leave=False):
            sequences.append(seq)
            labels.append(class_to_idx[label])
            flipped = augment_horizontal_flip(seq)
            sequences.append(flipped)
            labels.append(class_to_idx[label])
    return sequences, labels


def build_action_dataset():
    base_dir = Path(__file__).resolve().parents[1]
    raw_root = base_dir / "data" / "raw"
    ucf_root = raw_root / "ucf101"
    custom_root = raw_root / "custom"
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    sequences = []
    labels = []
    ucf_sequences, ucf_labels = load_ucf101_sequences(ucf_root, class_to_idx)
    sequences.extend(ucf_sequences)
    labels.extend(ucf_labels)
    custom_sequences, custom_labels = load_custom_sequences(custom_root, class_to_idx)
    sequences.extend(custom_sequences)
    labels.extend(custom_labels)
    if len(sequences) == 0:
        raise RuntimeError("No sequences were generated. Check that video files exist under data/raw.")
    X = np.stack(sequences, axis=0)
    y_idx = np.array(labels, dtype=np.int64)
    num_classes = len(CLASSES)
    y = np.eye(num_classes, dtype=np.float32)[y_idx]
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split_idx = int(0.8 * len(X))
    if split_idx == 0 or split_idx == len(X):
        raise RuntimeError("Not enough samples to create a non-empty train/test split.")
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = build_action_dataset()
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "action_X_train.npy", X_train)
    np.save(processed_dir / "action_y_train.npy", y_train)
    np.save(processed_dir / "action_X_test.npy", X_test)
    np.save(processed_dir / "action_y_test.npy", y_test)
    print("Saved processed datasets to", processed_dir)
    print("action_X_train.npy shape:", X_train.shape)
    print("action_y_train.npy shape:", y_train.shape)
    print("action_X_test.npy shape:", X_test.shape)
    print("action_y_test.npy shape:", y_test.shape)
