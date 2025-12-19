import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 48
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(CLASSES)


def _one_hot(labels: np.ndarray) -> np.ndarray:
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"Labels out of range [0, {NUM_CLASSES - 1}]: min={labels.min()}, max={labels.max}")
    return np.eye(NUM_CLASSES, dtype=np.float32)[labels]


def _collect_image_label_pairs(split_dir: Path, class_to_idx: dict):
    pairs = []
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    for class_name in CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        class_files = []
        for pattern in patterns:
            class_files.extend(class_dir.glob(pattern))
        class_files = sorted(class_files)

        label_idx = class_to_idx[class_name]
        for img_path in class_files:
            pairs.append((img_path, label_idx))

    return pairs


def _load_split(split_dir: Path, split_name: str, class_to_idx: dict):
    pairs = _collect_image_label_pairs(split_dir, class_to_idx)
    if len(pairs) == 0:
        raise RuntimeError(f"No images found in {split_dir}")

    X_list = []
    y_list = []
    skipped = 0

    for img_path, label_idx in tqdm(pairs, desc=f"Loading {split_name}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)

        X_list.append(img)
        y_list.append(label_idx)

    if len(X_list) == 0:
        raise RuntimeError(f"No valid images were loaded for split '{split_name}'.")

    if skipped > 0:
        print(f"⚠ Skipped {skipped} unreadable image(s) in split '{split_name}'.")

    X = np.stack(X_list, axis=0)
    y = _one_hot(np.array(y_list, dtype=np.int64))
    return X, y


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    raw_root = base_dir / "data" / "raw" / "fer2013"
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        print("❌ FER-2013 folder dataset not found.")
        print("Expected structure:")
        print(f"  {train_dir}/<class_name>/*.png")
        print(f"  {test_dir}/<class_name>/*.png")
        print("\nExpected class folders:")
        print("  " + ", ".join(CLASSES))
        sys.exit(1)

    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    X_train, y_train = _load_split(train_dir, "train", class_to_idx)
    X_test, y_test = _load_split(test_dir, "test", class_to_idx)

    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "emotion_X_train.npy", X_train)
    np.save(processed_dir / "emotion_y_train.npy", y_train)
    np.save(processed_dir / "emotion_X_test.npy", X_test)
    np.save(processed_dir / "emotion_y_test.npy", y_test)

    print("Saved processed emotion dataset to", processed_dir)
    print("emotion_X_train.npy shape:", X_train.shape)
    print("emotion_y_train.npy shape:", y_train.shape)
    print("emotion_X_test.npy shape:", X_test.shape)
    print("emotion_y_test.npy shape:", y_test.shape)
