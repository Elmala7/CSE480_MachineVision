import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Class order must match the one used in make_dataset_action.py
CLASSES = ["Walking", "Waving", "Standing", "Sitting"]


def load_processed_data():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"

    X_train_path = processed_dir / "action_X_train.npy"
    y_train_path = processed_dir / "action_y_train.npy"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Expected files: "
            f"{X_train_path.name}, {y_train_path.name} in {processed_dir}"
        )

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    return X_train, y_train


def pick_random_sample(X_train, y_train):
    if len(X_train) == 0:
        raise ValueError("Empty training set: no samples to inspect.")

    idx = np.random.randint(0, len(X_train))
    sample = X_train[idx]  # shape: (16, 128, 128, 3)
    label_one_hot = y_train[idx]

    if label_one_hot.ndim == 0 or label_one_hot.shape[0] != len(CLASSES):
        raise ValueError(
            f"Unexpected label shape {label_one_hot.shape}; "
            f"expected one-hot of length {len(CLASSES)}."
        )

    class_idx = int(np.argmax(label_one_hot))
    label_name = CLASSES[class_idx]

    print(f"Random sample index: {idx}")
    print(f"Label index: {class_idx}")
    print(f"Label name: {label_name}")

    return sample, label_name


def plot_sequence(sequence, label_name):
    if sequence.shape[0] != 16:
        raise ValueError(f"Expected sequence length 16, got {sequence.shape[0]}")

    # sequence shape: (16, H, W, 3), in BGR order from OpenCV
    # Convert each frame from BGR -> RGB for correct display with matplotlib
    frames_rgb = sequence[..., ::-1]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(16):
        ax = axes[i]
        ax.imshow(frames_rgb[i])
        ax.axis("off")
        ax.set_title(str(i + 1))

    fig.suptitle(f"Action: {label_name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train, y_train = load_processed_data()
    sample, label_name = pick_random_sample(X_train, y_train)
    print(f"Sample shape: {sample.shape}")
    plot_sequence(sample, label_name)
