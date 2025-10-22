"""Train a TensorFlow model using landmark sequences extracted from gesture videos."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from . import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a gesture recognition model with TensorFlow")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.FEATURES_DIR / "gesture_dataset.npz",
        help="Path to the .npz file containing arrays X and y.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=config.FEATURES_DIR / "gesture_dataset_labels.json",
        help="JSON file with the gesture to index mapping.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    return parser.parse_args()


def load_data(dataset_path: Path, validation_split: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return (X_train, y_train), (X_val, y_val)


def build_model(num_classes: int, sequence_length: int, feature_dim: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="landmarks")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main() -> None:
    args = parse_args()

    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, args.validation_split)
    sequence_length = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = int(np.max(np.concatenate([y_train, y_val])) + 1)

    model = build_model(num_classes, sequence_length, feature_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "checkpoint"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(config.LOGS_DIR / datetime.now().strftime("logs_%Y%m%d_%H%M%S"))),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = config.MODELS_DIR / f"gesture_model_{timestamp}"
    model.save(model_dir)
    print(f"💾 Modelo guardado en {model_dir}")

    history_path = model_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)
    print(f"📝 Historial de entrenamiento guardado en {history_path}")

    labels_dest = model_dir / "labels.json"
    labels_dest.write_text(args.labels.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"🗂️  Copia del mapa de etiquetas guardada en {labels_dest}")


if __name__ == "__main__":
    main()
