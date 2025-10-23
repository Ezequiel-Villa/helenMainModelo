"""Real-time gesture recognition using a TensorFlow model trained on video clips."""
# Ejecuta la cámara en vivo, detecta ambas manos con MediaPipe y clasifica la
# secuencia acumulada mediante el modelo entrenado en TensorFlow.
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from . import config
from .extract_landmarks import normalise_landmarks


def parse_args() -> argparse.Namespace:
    """Definir los parámetros de ejecución aceptados desde la terminal."""
    parser = argparse.ArgumentParser(description="Run real-time gesture detection")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the SavedModel produced by train_model.py",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional path to the labels.json file. If omitted, the script looks inside the model directory.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    return parser.parse_args()


def load_label_map(path: Path) -> Dict[int, str]:
    """Invertir el diccionario gesto->índice para mostrar etiquetas legibles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {idx: gesture for gesture, idx in data.items()}


def main() -> None:
    """Configurar MediaPipe, cargar el modelo y realizar inferencia cuadro a cuadro."""
    args = parse_args()

    label_path = args.labels or (args.model_dir / "labels.json")
    if not label_path.exists():
        raise FileNotFoundError("No se encontró labels.json. Indica la ruta con --labels.")

    idx_to_label = load_label_map(label_path)
    model = tf.keras.models.load_model(args.model_dir)

    # MediaPipe Hands detectará hasta dos manos y devolverá sus landmarks por frame.
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    buffer: Deque[np.ndarray] = deque(maxlen=args.sequence_length)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_features = np.zeros((config.MAX_HANDS, config.NUM_HAND_LANDMARKS, config.LANDMARK_DIM), dtype=np.float32)
            if results.multi_hand_landmarks and results.multi_handedness:
                ordering = {"Left": 0, "Right": 1}
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    idx = ordering.get(label, 0)
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                    frame_features[idx] = coords
                    drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            buffer.append(normalise_landmarks(frame_features.flatten()))

            if len(buffer) == args.sequence_length:
                # Cuando el buffer está lleno se construye el tensor y se predice la etiqueta.
                input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                probabilities = model.predict(input_tensor, verbose=0)[0]
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                if confidence >= args.confidence_threshold:
                    label = idx_to_label.get(pred_idx, "?")
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

            cv2.imshow("Detección de gestos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
