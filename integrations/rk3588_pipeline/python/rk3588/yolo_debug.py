from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2  # type: ignore[import-not-found]


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from common.camera import LatestFrameGrabber, split_stitched_frame
from common.config import load_app_config
from common.rectify import StereoRectifier
from rk3588.yolo_backend import RknnLiteYoloBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 YOLO debug viewer")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "runtime" / "rk3588_vector_fast.json"),
        help="Path to vector runtime config JSON",
    )
    return parser.parse_args()


def draw_detections(image, detections, color):
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    if config.hand_model is None or config.target_model is None:
        raise ValueError("Config must include hand_model and target_model")

    grabber = LatestFrameGrabber(config.capture)
    rectifier = StereoRectifier.from_config(config.rectify)
    hand_backend = RknnLiteYoloBackend(config.hand_model)
    target_backend = RknnLiteYoloBackend(config.target_model)

    cv2.namedWindow("RK3588 YOLO Debug", cv2.WINDOW_NORMAL)
    grabber.start()
    last_log = time.perf_counter()

    try:
        while True:
            packet = grabber.read(config.runtime.capture_timeout_ms / 1000.0)
            if packet is None:
                continue

            left_bgr, right_bgr = split_stitched_frame(packet.frame, config.capture)
            left_bgr, _ = rectifier.apply(left_bgr, right_bgr)

            hand = hand_backend.infer(left_bgr)
            target = target_backend.infer(left_bgr)

            frame = draw_detections(left_bgr, target, (0, 255, 255))
            frame = draw_detections(frame, hand, (0, 255, 0))

            now = time.perf_counter()
            if now - last_log >= 1.0:
                top_target = [f"{d.class_name}:{d.confidence:.2f}" for d in target[:5]]
                top_hand = [f"{d.class_name}:{d.confidence:.2f}" for d in hand[:5]]
                print(
                    {
                        "frame": packet.index,
                        "hand_count": len(hand),
                        "target_count": len(target),
                        "top_hand": top_hand,
                        "top_target": top_target,
                    },
                    flush=True,
                )
                last_log = now

            cv2.imshow("RK3588 YOLO Debug", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
