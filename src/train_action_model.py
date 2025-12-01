import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD, Adam, Adagrad

# Dataset / model configuration
IMG_SIZE = 128
SEQ_LENGTH = 16
NUM_CLASSES = 4
BATCH_SIZE = 4
EPOCHS = 15
OPTIMIZERS = ["SGD", "Adam", "Adagrad"]


def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"

    X_train = np.load(processed_dir / "action_X_train.npy")
    y_train = np.load(processed_dir / "action_y_train.npy")
    X_test = np.load(processed_dir / "action_X_test.npy")
    y_test = np.load(processed_dir / "action_y_test.npy")

    return X_train, y_train, X_test, y_test


def build_model(optimizer_name: str) -> models.Model:
    """Builds a CNN-LSTM action recognition model with MobileNetV2 backbone.

    Input: sequence of 16 frames with shape (16, 128, 128, 3).
    Spatial features: MobileNetV2 (ImageNet, include_top=False, pooling='avg').
    Temporal features: LSTM(64, dropout=0.3).
    Classifier: Dense(4, softmax).
    """

    inputs = layers.Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3), name="frames")

    # MobileNetV2 expects inputs in [-1, 1]; our dataset is in [0, 1]
    x = layers.Lambda(lambda z: z * 2.0 - 1.0, name="scale_minus1_1")(inputs)

    base_cnn = MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Freeze backbone for faster training on M1
    base_cnn.trainable = False

    x = layers.TimeDistributed(base_cnn, name="frame_cnn")(x)
    x = layers.LSTM(64, dropout=0.3, name="lstm")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"action_model_{optimizer_name.lower()}")

    opt_name = optimizer_name.lower()
    if opt_name == "sgd":
        optimizer = SGD(learning_rate=1e-3, momentum=0.9)
    elif opt_name == "adam":
        optimizer = Adam(learning_rate=1e-4)
    elif opt_name == "adagrad":
        optimizer = Adagrad(learning_rate=1e-3)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_experiments():
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    history_dict = {}
    results = []

    for opt_name in OPTIMIZERS:
        print("\n" + "=" * 60)
        print(f"Training with optimizer: {opt_name}")
        print("=" * 60)

        model = build_model(opt_name)
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        history_dict[opt_name] = history.history

        model_path = models_dir / f"action_model_{opt_name.lower()}.keras"
        model.save(model_path)
        print(f"Saved model to {model_path}")

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{opt_name} Test Accuracy: {test_acc:.4f}")

        results.append({
            "optimizer": opt_name,
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        })

    # Visualization: accuracy & validation loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for opt_name in OPTIMIZERS:
        hist = history_dict[opt_name]
        # Handle possible key naming differences
        if "accuracy" in hist:
            train_acc = hist["accuracy"]
        elif "categorical_accuracy" in hist:
            train_acc = hist["categorical_accuracy"]
        else:
            raise KeyError(f"No accuracy key found in history for {opt_name}: {hist.keys()}")

        val_loss = hist.get("val_loss")
        epochs_range = range(1, len(train_acc) + 1)

        ax1.plot(epochs_range, train_acc, label=opt_name)
        if val_loss is not None:
            ax2.plot(epochs_range, val_loss, label=opt_name)

    ax1.set_title("Training Accuracy vs Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.set_title("Validation Loss vs Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    plot_path = reports_dir / "milestone1_optimizer_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved optimizer comparison plot to {plot_path}")

    # Final summary table
    print("\nFinal Test Accuracy by Optimizer:")
    print("-" * 40)
    print(f"{'Optimizer':<12}{'Test Accuracy':>15}")
    print("-" * 40)
    for r in results:
        print(f"{r['optimizer']:<12}{r['test_accuracy']:>15.4f}")
    print("-" * 40)


if __name__ == "__main__":
    # Limit TensorFlow logging noise
    tf.get_logger().setLevel("ERROR")
    run_experiments()
