from __future__ import annotations

import csv
import json
import signal
import time
import os
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.camera import LatestFrameGrabber, split_stitched_frame
from common.config import AppConfig
from common.preprocess import StereoPreprocessor, restore_disparity
from common.rectify import StereoRectifier
from common.visualize import disparity_to_depth
from rk3588.rknn_backend import RknnLiteStereoBackend
from rk3588.yolo_backend import Detection, RknnLiteYoloBackend


@dataclass
class ObstacleWriter:
    csv_path: Path | None
    jsonl_path: Path | None
    rolling_path: Path | None
    preview_path: Path | None

    def __post_init__(self) -> None:
        self.csv_file = None
        self.csv_writer = None
        self.jsonl_file = None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = self.csv_path.open('a', encoding='utf-8', newline='')
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    'timestamp_s', 'frame_index', 'status', 'obstacle_class_name',
                    'distance_m', 'vector_x_m', 'vector_z_m',
                    'center_x_px', 'center_y_px',
                    'bbox_xyxy',
                ],
            )
            if self.csv_path.stat().st_size == 0:
                self.csv_writer.writeheader()
                self.csv_file.flush()
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = self.jsonl_path.open('a', encoding='utf-8')

    def write(self, payload: dict[str, object]) -> None:
        if self.csv_writer and self.csv_file:
            self.csv_writer.writerow(payload)
            self.csv_file.flush()
        if self.jsonl_file:
            self.jsonl_file.write(json.dumps(payload, ensure_ascii=False) + '\n')
            self.jsonl_file.flush()
        if self.rolling_path:
            self.rolling_path.parent.mkdir(parents=True, exist_ok=True)
            self.rolling_path.write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')

    def write_preview(self, image: np.ndarray) -> None:
        if not self.preview_path:
            return
        self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        tmp_path = self.preview_path.with_suffix(self.preview_path.suffix + '.tmp')
        tmp_path.write_bytes(encoded.tobytes())
        os.replace(tmp_path, self.preview_path)

    def close(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        if self.jsonl_file:
            self.jsonl_file.close()


class StereoObstacleRuntime:
    def __init__(self, config: AppConfig) -> None:
        if config.target_model is None:
            raise ValueError('Obstacle runtime requires target_model config')
        self.config = config
        self.stereo_backend = RknnLiteStereoBackend(config)
        self.target_backend = RknnLiteYoloBackend(config.target_model)
        self.preprocessor = StereoPreprocessor(config.model)
        self.rectifier = StereoRectifier.from_config(config.rectify)
        self.grabber = LatestFrameGrabber(config.capture)
        self._interrupted = False
        output_cfg = config.obstacle_output
        if output_cfg is not None and (
            output_cfg.output_csv_path
            or output_cfg.output_jsonl_path
            or output_cfg.rolling_output_jsonl_path
            or output_cfg.preview_image_path
        ):
            csv_path = Path(output_cfg.output_csv_path) if output_cfg.output_csv_path else None
            jsonl_path = Path(output_cfg.output_jsonl_path) if output_cfg.output_jsonl_path else None
            rolling_path = Path(output_cfg.rolling_output_jsonl_path) if output_cfg.rolling_output_jsonl_path else None
            preview_path = Path(output_cfg.preview_image_path) if output_cfg.preview_image_path else None
        else:
            runtime_dir = Path(config.config_path).resolve().parents[3] / 'outputs' / 'obstacle_mode'
            csv_path = runtime_dir / 'obstacle_vector.csv'
            jsonl_path = runtime_dir / 'obstacle_vector.jsonl'
            rolling_path = runtime_dir / 'obstacle_vector_latest.jsonl'
            preview_path = runtime_dir / 'obstacle_preview.jpg'
        self.writer = ObstacleWriter(
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            rolling_path=rolling_path,
            preview_path=preview_path,
        )
        self._last_emit_ts = 0.0
        self._smoothed_disparity = None
        self._last_no_frame_emit_ts = 0.0

    def _handle_sigint(self, signum: int, frame: object) -> None:
        self._interrupted = True

    def _stabilize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        alpha = min(max(float(self.config.runtime.temporal_smoothing_alpha), 0.0), 1.0)
        if alpha <= 0.0:
            return disparity
        if self._smoothed_disparity is None or self._smoothed_disparity.shape != disparity.shape:
            self._smoothed_disparity = disparity.copy()
            return disparity
        valid = disparity > self.config.depth.min_valid_disp
        self._smoothed_disparity[valid] = alpha * disparity[valid] + (1.0 - alpha) * self._smoothed_disparity[valid]
        return self._smoothed_disparity.copy()

    def _sample_depth(self, depth_map: np.ndarray, center_xy: tuple[float, float], region_px: int = 9) -> float | None:
        x = int(round(center_xy[0]))
        y = int(round(center_xy[1]))
        radius = max(0, region_px // 2)
        h, w = depth_map.shape[:2]
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _pixel_to_xz(self, center_xy: tuple[float, float], depth_m: float, image_shape: tuple[int, int]) -> tuple[float, float]:
        h, w = image_shape
        cx = w / 2.0
        fx = float(self.config.depth.focal_px)
        x_m = ((float(center_xy[0]) - cx) * depth_m) / fx
        return x_m, depth_m

    def _nearest_obstacle(self, detections: list[Detection], depth_map: np.ndarray, image_shape: tuple[int, int]) -> tuple[Detection | None, float | None, tuple[float, float] | None]:
        best = None
        for det in detections:
            depth_m = self._sample_depth(depth_map, det.center_xy)
            if depth_m is None:
                continue
            x_m, z_m = self._pixel_to_xz(det.center_xy, depth_m, image_shape)
            score = np.hypot(x_m, z_m)
            if best is None or score < best[0]:
                best = (score, det, depth_m, (x_m, z_m))
        if best is None:
            return None, None, None
        return best[1], best[2], best[3]

    def _draw_preview(self, left_bgr: np.ndarray, depth_map: np.ndarray, chosen: Detection | None, vector_xz: tuple[float, float] | None, distance_m: float | None) -> np.ndarray:
        overlay = left_bgr.copy()
        if chosen is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in chosen.bbox_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
            label = f'{chosen.class_name} {chosen.confidence:.2f}'
            if distance_m is not None:
                label += f' {distance_m:.2f}m'
            cv2.putText(overlay, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        if vector_xz is not None:
            text = f'nearest vec(x,z)=({vector_xz[0]:.3f}, {vector_xz[1]:.3f}) m'
            cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2, cv2.LINE_AA)
        else:
            cv2.putText(overlay, 'YOLO preview: waiting for obstacle', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2, cv2.LINE_AA)
        return overlay

    def run(self) -> None:
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.grabber.start()
        fallback_interval = max(1, int(self.config.runtime.yolo_infer_interval))
        last_detections: list[Detection] = []
        try:
            while True:
                if self._interrupted:
                    break
                packet = self.grabber.read(self.config.runtime.capture_timeout_ms / 1000.0)
                if packet is None:
                    now = time.time()
                    if now - self._last_no_frame_emit_ts >= 1.0:
                        self._last_no_frame_emit_ts = now
                        payload = {
                            'timestamp_s': round(now, 6),
                            'frame_index': -1,
                            'status': 'no_frame',
                            'obstacle_class_name': None,
                            'distance_m': None,
                            'vector_x_m': None,
                            'vector_z_m': None,
                            'center_x_px': None,
                            'center_y_px': None,
                            'bbox_xyxy': None,
                        }
                        self.writer.write(payload)
                        print(json.dumps(payload, ensure_ascii=False), flush=True)
                    continue

                left_bgr, right_bgr = split_stitched_frame(packet.frame, self.config.capture)
                left_bgr, right_bgr = self.rectifier.apply(left_bgr, right_bgr)
                prepared = self.preprocessor.prepare(left_bgr, right_bgr)
                stereo_raw = self.stereo_backend.infer(prepared.left, prepared.right)
                disparity = restore_disparity(stereo_raw, prepared)
                disparity = self._stabilize_disparity(disparity)
                depth_map = disparity_to_depth(disparity, self.config.depth)

                if packet.index == 0 or packet.index % fallback_interval == 0:
                    last_detections = self.target_backend.infer(left_bgr)

                chosen, depth_m, vector_xz = self._nearest_obstacle(last_detections, depth_map, left_bgr.shape[:2])
                distance_m = None if vector_xz is None else float(np.hypot(vector_xz[0], vector_xz[1]))
                status = 'no_obstacle'
                if chosen is not None and vector_xz is not None and depth_m is not None:
                    status = 'tracking'

                preview = self._draw_preview(left_bgr, depth_map, chosen, vector_xz, distance_m)
                self.writer.write_preview(preview)

                now = time.time()
                if now - self._last_emit_ts < 0.5:
                    continue
                self._last_emit_ts = now

                payload = {
                    'timestamp_s': round(now, 6),
                    'frame_index': packet.index,
                    'status': status,
                    'obstacle_class_name': chosen.class_name if chosen else None,
                    'distance_m': round(distance_m, 6) if distance_m is not None else None,
                    'vector_x_m': round(vector_xz[0], 6) if vector_xz is not None else None,
                    'vector_z_m': round(vector_xz[1], 6) if vector_xz is not None else None,
                    'center_x_px': round(chosen.center_xy[0], 2) if chosen else None,
                    'center_y_px': round(chosen.center_xy[1], 2) if chosen else None,
                    'bbox_xyxy': [round(v, 2) for v in chosen.bbox_xyxy] if chosen else None,
                }
                self.writer.write(payload)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.grabber.stop()
            self.writer.close()
            cv2.destroyAllWindows()
