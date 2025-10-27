"""Utility for recording short gesture clips with both hands visible.

Press ``s`` to start capturing a clip and ``q`` to exit. Each clip is saved to
``data/raw_videos/<gesture>/<gesture>_<timestamp>.mp4`` so it can later be
processed into landmarks. Recording defaults are defined in :mod:`config`.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2

#from . import config
import config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record gesture clips with OpenCV")
    parser.add_argument(
        "gesture",
        help="Name of the gesture being recorded (used as folder prefix).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=config.CLIP_DURATION,
        help="Length of each recorded clip in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.FPS,
        help="Frames per second of the recorded clip.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index passed to OpenCV (default: 0).",
    )
    return parser.parse_args()


def record_clip(cap: cv2.VideoCapture, writer: cv2.VideoWriter, duration: float) -> None:
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cv2.imshow("Grabando gesto", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    args = parse_args()

    gesture_dir = config.VIDEOS_DIR / args.gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")

    print(f"Grabando gesto '{args.gesture}'. Presiona 's' para capturar un clip, 'q' para salir.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Vista previa", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = gesture_dir / f"{args.gesture}_{timestamp}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    args.fps,
                    (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                )
                print(f"➡️  Grabando clip en {video_path}")
                record_clip(cap, writer, args.duration)
                writer.release()
                print("✅ Clip guardado\n")

            elif key == ord("q"):
                print("Grabación finalizada por el usuario.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
