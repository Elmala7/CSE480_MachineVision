import numpy as np
from pathlib import Path

import tensorflow as tf

CLASSES = ["Walking", "Waving", "Standing", "Sitting"]


def list_keras_files(models_dir: Path):
    print(f"Searching for .keras files in: {models_dir}")
    if not models_dir.exists():
        print("models directory does not exist.")
        return []

    keras_files = sorted(models_dir.glob("*.keras"))
    if not keras_files:
        print("No .keras files found.")
        return []

    print("Found .keras model files:")
    for path in keras_files:
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f" - {path.name}: {size_mb:.2f} MB ({size_bytes} bytes)")

    return keras_files


def load_any_model(models_dir: Path):
    adam_path = models_dir / "action_model_adam.keras"
    adagrad_path = models_dir / "action_model_adagrad.keras"

    model_path = None
    if adam_path.exists():
        model_path = adam_path
    elif adagrad_path.exists():
        model_path = adagrad_path
    else:
        # Fallback: pick the first .keras file if available
        keras_files = sorted(models_dir.glob("*.keras"))
        if keras_files:
            model_path = keras_files[0]

    if model_path is None:
        raise FileNotFoundError("No suitable .keras model file found in models/ directory.")

    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    return model, model_path


def load_random_test_sample(processed_dir: Path):
    x_test_path = processed_dir / "action_X_test.npy"
    if not x_test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {x_test_path}")

    X_test = np.load(x_test_path)
    if len(X_test) == 0:
        raise ValueError("Test set is empty; cannot pick a random sample.")

    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx: idx + 1]  # keep batch dimension for model.predict
    print(f"\nUsing random test sample index: {idx}")
    print(f"Sample shape: {sample.shape}")
    return sample, idx


def main():
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"

    # 1) List all .keras files and sizes
    list_keras_files(models_dir)

    # 2) Load preferred model (Adam, then Adagrad)
    model, model_path = load_any_model(models_dir)

    print("\nModel summary:")
    model.summary()

    # 3) Load one random test sample
    sample, idx = load_random_test_sample(processed_dir)

    # 4) Run inference
    print("\nRunning model.predict on the sample...")
    probs = model.predict(sample)

    if probs.ndim != 2 or probs.shape[1] != len(CLASSES):
        raise ValueError(
            f"Unexpected prediction shape {probs.shape}; expected (1, {len(CLASSES)})."
        )

    probs_row = probs[0]
    print("Raw output probabilities:")
    for i, (cls, p) in enumerate(zip(CLASSES, probs_row)):
        print(f"  Class {i} ({cls}): {p:.4f}")

    pred_idx = int(np.argmax(probs_row))
    pred_class = CLASSES[pred_idx]

    print(f"\nPredicted class index: {pred_idx}")
    print(f"Predicted class name: {pred_class}")


if __name__ == "__main__":
    # Reduce TensorFlow logging noise
    tf.get_logger().setLevel("ERROR")
    main()
