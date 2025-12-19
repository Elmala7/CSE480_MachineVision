import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 15


def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"

    X_train = np.load(processed_dir / "emotion_X_train.npy")
    y_train = np.load(processed_dir / "emotion_y_train.npy")
    X_test = np.load(processed_dir / "emotion_X_test.npy")
    y_test = np.load(processed_dir / "emotion_y_test.npy")

    return X_train, y_train, X_test, y_test


def build_vgg_model() -> models.Model:
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="emotion_simple_vgg")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _residual_block(x, filters: int, stride: int = 1):
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet_model() -> models.Model:
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = _residual_block(x, 32, stride=1)
    x = _residual_block(x, 32, stride=1)

    x = _residual_block(x, 64, stride=2)
    x = _residual_block(x, 64, stride=1)

    x = _residual_block(x, 128, stride=2)
    x = _residual_block(x, 128, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="emotion_mini_resnet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _get_metric(history: dict, keys):
    for k in keys:
        if k in history:
            return history[k]
    raise KeyError(f"None of the keys {keys} found in history. Available keys: {list(history.keys())}")


def _train_one(name: str, build_fn, X_train, y_train, X_val, y_val, models_dir: Path):
    tf.keras.backend.clear_session()

    model = build_fn()
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)
    model.summary()

    weights_path = models_dir / f"emotion_{name}_best.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint],
    )

    model.load_weights(weights_path)
    val_acc = _get_metric(history.history, ["val_accuracy", "val_categorical_accuracy"])
    best_val_acc = float(np.max(val_acc))

    val_loss, val_acc_eval = model.evaluate(X_val, y_val, verbose=0)
    print(f"{name} best val_accuracy (history): {best_val_acc:.4f}")
    print(f"{name} val_accuracy (evaluate best weights): {val_acc_eval:.4f}")

    return model, history.history, best_val_acc


def run_experiments():
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    print("Loaded emotion dataset:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    vgg_model, vgg_hist, vgg_best = _train_one(
        name="vgg",
        build_fn=build_vgg_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        models_dir=models_dir,
    )

    resnet_model, resnet_hist, resnet_best = _train_one(
        name="mini_resnet",
        build_fn=build_resnet_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        models_dir=models_dir,
    )

    vgg_val_acc = _get_metric(vgg_hist, ["val_accuracy", "val_categorical_accuracy"])
    resnet_val_acc = _get_metric(resnet_hist, ["val_accuracy", "val_categorical_accuracy"])

    epochs_range = range(1, len(vgg_val_acc) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, vgg_val_acc, label="Model A - Simple VGG")
    ax.plot(epochs_range, resnet_val_acc, label="Model B - Mini-ResNet")
    ax.set_title("FER-2013 Architecture Experiment: Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend()
    fig.tight_layout()

    plot_path = reports_dir / "milestone2_architecture_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved architecture comparison plot to {plot_path}")

    if resnet_best > vgg_best:
        best_name = "mini_resnet"
        best_model = resnet_model
        best_score = resnet_best
    else:
        best_name = "vgg"
        best_model = vgg_model
        best_score = vgg_best

    best_path = models_dir / "emotion_model_best.keras"
    best_model.save(best_path)
    print(f"Saved best model ({best_name}, best val_accuracy={best_score:.4f}) to {best_path}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    run_experiments()
