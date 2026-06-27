from __future__ import annotations

import csv
import json
import time
import threading
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import signal

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.camera import LatestFrameGrabber, split_stitched_frame
from common.config import AppConfig, VectorConfig
from common.preprocess import StereoPreprocessor, restore_disparity
from common.rectify import StereoRectifier
from common.visualize import build_preview, colorize_depth, colorize_disparity, disparity_to_depth
from rk3588.rknn_backend import RknnLiteStereoBackend
from rk3588.yolo_backend import Detection, RknnLiteYoloBackend


@dataclass
class VectorState:
    hand_xyz: tuple[float, float, float] | None
    target_xyz: tuple[float, float, float] | None
    vector_xyz: tuple[float, float, float] | None
    distance_m: float | None
    status: str
    vector_active: bool


_TRACKING_HOLD_FRAMES = 2
_REFERENCE_WEIGHT = 0.35


def _sample_point_depth(depth_map: np.ndarray, x: int, y: int, region_px: int) -> float | None:
    radius = max(0, int(region_px) // 2)
    h, w = depth_map.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _pixel_to_xyz(x: float, y: float, z: float, cx: float, cy: float, fx: float, fy: float) -> tuple[float, float, float]:
    return ((x - cx) * z / fx, (y - cy) * z / fy, z)


def _pick_best(detections: list[Detection], class_name: str) -> Detection | None:
    matched = [item for item in detections if item.class_name == class_name]
    if not matched:
        return None
    matched.sort(key=lambda item: item.confidence, reverse=True)
    return matched[0]


def _pick_with_reference(
    detections: list[Detection],
    class_name: str,
    reference_xy: tuple[float, float] | None,
    image_shape: tuple[int, int] | None = None,
) -> Detection | None:
    matched = [item for item in detections if item.class_name == class_name]
    if not matched:
        return None
    if reference_xy is None or image_shape is None:
        return _pick_best(detections, class_name)

    height, width = image_shape
    diag = max(float(np.hypot(width, height)), 1.0)

    def score(item: Detection) -> float:
        dx = item.center_xy[0] - reference_xy[0]
        dy = item.center_xy[1] - reference_xy[1]
        normalized_distance = float(np.hypot(dx, dy)) / diag
        return float(item.confidence) - (_REFERENCE_WEIGHT * normalized_distance)

    matched.sort(key=score, reverse=True)
    return matched[0]


class CsvJsonlWriter:
    def __init__(self, config: VectorConfig) -> None:
        self.config = config
        self.csv_file = None
        self.csv_writer = None
        self.jsonl_file = None
        self.rolling_jsonl_path: Path | None = None
        self.preview_image_path: Path | None = None
        if config.output_csv_path:
            path = Path(config.output_csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = path.open("a", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._fieldnames())
            if path.stat().st_size == 0:
                self.csv_writer.writeheader()
                self.csv_file.flush()
        if config.output_jsonl_path:
            path = Path(config.output_jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_file = path.open("a", encoding="utf-8")
        if config.rolling_output_jsonl_path:
            path = Path(config.rolling_output_jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.rolling_jsonl_path = path
        if config.preview_image_path:
            path = Path(config.preview_image_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.preview_image_path = path

    def _fieldnames(self) -> list[str]:
        return [
            "timestamp_s",
            "frame_index",
            "status",
            "vector_active",
            "hand_x_m",
            "hand_y_m",
            "hand_z_m",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "vector_x_m",
            "vector_y_m",
            "vector_z_m",
            "distance_m",
        ]

    def write(self, payload: dict[str, object]) -> None:
        if self.csv_writer is not None and self.csv_file is not None:
            self.csv_writer.writerow(payload)
            self.csv_file.flush()
        if self.jsonl_file is not None:
            self.jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.jsonl_file.flush()

    def write_rolling(self, payloads: list[dict[str, object]]) -> None:
        if self.rolling_jsonl_path is None:
            return
        with self.rolling_jsonl_path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_preview(self, image: np.ndarray) -> None:
        if self.preview_image_path is None:
            return
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        tmp_path = self.preview_image_path.with_suffix(self.preview_image_path.suffix + '.tmp')
        tmp_path.write_bytes(encoded.tobytes())
        os.replace(tmp_path, self.preview_image_path)

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
        if self.jsonl_file is not None:
            self.jsonl_file.close()


class LivePreviewWriter:
    def __init__(
        self,
        preview_path: Path | None,
        capture_config,
        max_fps: float = 12.0,
        jpeg_quality: int = 65,
        output_scale: float = 0.5,
    ) -> None:
        self.preview_path = preview_path
        self.capture_config = capture_config
        self.max_interval_s = 1.0 / max(max_fps, 1.0)
        self.jpeg_quality = max(40, min(90, int(jpeg_quality)))
        self.output_scale = min(max(float(output_scale), 0.2), 1.0)
        self._latest_frame: np.ndarray | None = None
        self._latest_index = -1
        self._last_written_index = -1
        self._last_write_ts = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._hand_detections: list[Detection] = []
        self._target_detections: list[Detection] = []

    def start(self) -> None:
        if self.preview_path is None or self._running:
            return
        self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='live-preview-writer')
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def submit(self, packet) -> None:
        if self.preview_path is None:
            return
        with self._lock:
            self._latest_frame = packet.frame.copy()
            self._latest_index = int(packet.index)
        self._event.set()

    def update_detections(self, hand_detections: list[Detection], target_detections: list[Detection]) -> None:
        with self._lock:
            self._hand_detections = list(hand_detections or [])
            self._target_detections = list(target_detections or [])

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return canvas

    def _loop(self) -> None:
        while self._running:
            self._event.wait(timeout=0.2)
            self._event.clear()
            if not self._running:
                break
            now = time.perf_counter()
            if now - self._last_write_ts < self.max_interval_s:
                continue
            with self._lock:
                frame = self._latest_frame
                frame_index = self._latest_index
                hand_detections = list(self._hand_detections)
                target_detections = list(self._target_detections)
            if frame is None or frame_index == self._last_written_index:
                continue
            try:
                left_bgr, _ = split_stitched_frame(frame, self.capture_config)
                preview = self._draw_detections(left_bgr, target_detections, (0, 255, 255))
                preview = self._draw_detections(preview, hand_detections, (0, 255, 0))
                if self.output_scale < 0.999:
                    preview = cv2.resize(
                        preview,
                        (
                            max(1, int(round(preview.shape[1] * self.output_scale))),
                            max(1, int(round(preview.shape[0] * self.output_scale))),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    continue
                tmp_path = self.preview_path.with_suffix(self.preview_path.suffix + '.tmp')
                tmp_path.write_bytes(encoded.tobytes())
                os.replace(tmp_path, self.preview_path)
                self._last_written_index = frame_index
                self._last_write_ts = now
            except Exception:
                pass


class StereoVectorRuntime:
    def __init__(self, config: AppConfig) -> None:
        if config.hand_model is None or config.target_model is None or config.vector is None:
            raise ValueError("Unified vector runtime requires hand_model, target_model and vector config")
        self.config = config
        self.vector_config = config.vector
        self.stereo_backend = RknnLiteStereoBackend(config)
        self.hand_backend = RknnLiteYoloBackend(config.hand_model)
        self.target_backend = RknnLiteYoloBackend(config.target_model)
        self.preprocessor = StereoPreprocessor(config.model)
        self.rectifier = StereoRectifier.from_config(config.rectify)
        self.grabber = LatestFrameGrabber(config.capture)
        self._smoothed_disparity: np.ndarray | None = None
        self._smoothed_vector: np.ndarray | None = None
        self._cached_hand_detections: list[Detection] = []
        self._cached_target_detections: list[Detection] = []
        self._last_hand_detection: Detection | None = None
        self._last_target_detection: Detection | None = None
        self._hand_missing_frames = 0
        self._target_missing_frames = 0
        self._interrupted = False
        self._active_target_class_name = self.vector_config.target_class_name
        self.writer = CsvJsonlWriter(self.vector_config)
        self.live_preview = LivePreviewWriter(
            self.writer.preview_image_path,
            config.capture,
            max_fps=20.0,
            jpeg_quality=60,
            output_scale=0.5,
        )
        self.grabber.add_frame_listener(self.live_preview.submit)
        self.recent_payloads: deque[dict[str, object]] = deque()

    def _handle_sigint(self, signum: int, frame: object) -> None:
        self._interrupted = True

    def _select_hand_target(
        self,
        left_bgr: np.ndarray,
        hand_detections: list[Detection],
        target_detections: list[Detection],
    ) -> tuple[Detection | None, Detection | None]:
        h, w = left_bgr.shape[:2]
        center_xy = (w * 0.5, h * 0.5)

        hand_reference = self._last_hand_detection.center_xy if self._last_hand_detection is not None else center_xy
        hand = _pick_with_reference(hand_detections, self.vector_config.hand_class_name, hand_reference, (h, w))
        if hand is not None:
            self._last_hand_detection = hand
            self._hand_missing_frames = 0
        elif self._last_hand_detection is not None and self._hand_missing_frames < _TRACKING_HOLD_FRAMES:
            hand = self._last_hand_detection
            self._hand_missing_frames += 1
        else:
            self._last_hand_detection = None
            self._hand_missing_frames = 0

        target_reference = (
            hand.center_xy
            if hand is not None
            else (self._last_target_detection.center_xy if self._last_target_detection is not None else center_xy)
        )
        target = _pick_with_reference(
            target_detections,
            self._active_target_class_name,
            target_reference,
            (h, w),
        )
        if target is not None:
            self._last_target_detection = target
            self._target_missing_frames = 0
        elif self._last_target_detection is not None and self._target_missing_frames < _TRACKING_HOLD_FRAMES:
            target = self._last_target_detection
            self._target_missing_frames += 1
        else:
            self._last_target_detection = None
            self._target_missing_frames = 0

        return hand, target

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

    def _smooth_vector(self, vector_xyz: np.ndarray) -> np.ndarray:
        alpha = min(max(float(self.vector_config.smoothing_alpha), 0.0), 1.0)
        if alpha <= 0.0:
            return vector_xyz
        if self._smoothed_vector is None or self._smoothed_vector.shape != vector_xyz.shape:
            self._smoothed_vector = vector_xyz.copy()
            return vector_xyz
        self._smoothed_vector = alpha * vector_xyz + (1.0 - alpha) * self._smoothed_vector
        return self._smoothed_vector.copy()

    def _compute_vector_state(
        self,
        left_bgr: np.ndarray,
        depth_map: np.ndarray,
        hand_detections: list[Detection],
        target_detections: list[Detection],
    ) -> VectorState:
        hand, target = self._select_hand_target(left_bgr, hand_detections, target_detections)
        if hand is None and target is None:
            return VectorState(None, None, None, None, "no_hand_no_target", False)
        if hand is None:
            return VectorState(None, None, None, None, "no_hand", False)
        if target is None:
            return VectorState(None, None, None, None, "no_target", False)

        h, w = left_bgr.shape[:2]
        cx = w / 2.0
        cy = h / 2.0
        fx = float(self.config.depth.focal_px)
        fy = float(self.config.depth.focal_px)

        hand_x_px = int(round(hand.center_xy[0]))
        hand_y_px = int(round(hand.center_xy[1]))
        target_x_px = int(round(target.center_xy[0]))
        target_y_px = int(round(target.center_xy[1]))

        hand_z = _sample_point_depth(depth_map, hand_x_px, hand_y_px, self.vector_config.sample_region_px)
        target_z = _sample_point_depth(depth_map, target_x_px, target_y_px, self.vector_config.sample_region_px)
        if hand_z is None or target_z is None:
            return VectorState(None, None, None, None, "invalid_depth", False)

        hand_xyz = _pixel_to_xyz(float(hand_x_px), float(hand_y_px), hand_z, cx, cy, fx, fy)
        target_xyz = _pixel_to_xyz(float(target_x_px), float(target_y_px), target_z, cx, cy, fx, fy)
        vector_xyz = np.asarray(target_xyz, dtype=np.float32) - np.asarray(hand_xyz, dtype=np.float32)
        vector_xyz = self._smooth_vector(vector_xyz)
        distance_m = float(np.linalg.norm(vector_xyz))
        if distance_m <= float(self.vector_config.stop_distance_m):
            return VectorState(hand_xyz, target_xyz, (0.0, 0.0, 0.0), distance_m, "stop", False)
        return VectorState(hand_xyz, target_xyz, tuple(float(v) for v in vector_xyz), distance_m, "tracking", True)

    def _reload_target_class(self, frame_index: int) -> None:
        path_str = self.vector_config.target_input_path
        if not path_str:
            return
        interval = max(1, int(self.vector_config.target_reload_interval_frames))
        if frame_index != 0 and frame_index % interval != 0:
            return

        path = Path(path_str)
        if not path.exists():
            return

        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not raw:
            return

        target_name: str | None = None
        if self.vector_config.target_input_format.lower() == "text":
            target_name = raw
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return
            if isinstance(data, dict):
                value = data.get(self.vector_config.target_input_key)
                if isinstance(value, str):
                    target_name = value

        if not target_name:
            return

        target_name = target_name.strip()
        if not target_name or target_name == self._active_target_class_name:
            return

        self._active_target_class_name = target_name
        self._cached_target_detections = []
        self._last_target_detection = None
        self._target_missing_frames = 0

    def _append_recent(self, payload: dict[str, object]) -> None:
        now = float(payload["timestamp_s"])
        self.recent_payloads.append(payload)
        window_s = float(self.vector_config.max_history_seconds)
        while self.recent_payloads and now - float(self.recent_payloads[0]["timestamp_s"]) > window_s:
            self.recent_payloads.popleft()
        self.writer.write_rolling(list(self.recent_payloads))

    def _build_payload(self, frame_index: int, vector_state: VectorState) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestamp_s": round(time.time(), 6),
            "frame_index": frame_index,
            "status": vector_state.status,
            "vector_active": vector_state.vector_active,
            "hand_x_m": None,
            "hand_y_m": None,
            "hand_z_m": None,
            "target_x_m": None,
            "target_y_m": None,
            "target_z_m": None,
            "vector_x_m": None,
            "vector_y_m": None,
            "vector_z_m": None,
            "distance_m": None,
        }
        if vector_state.hand_xyz is not None:
            payload["hand_x_m"], payload["hand_y_m"], payload["hand_z_m"] = [round(v, 6) for v in vector_state.hand_xyz]
        if vector_state.target_xyz is not None:
            payload["target_x_m"], payload["target_y_m"], payload["target_z_m"] = [round(v, 6) for v in vector_state.target_xyz]
        if vector_state.vector_xyz is not None:
            payload["vector_x_m"], payload["vector_y_m"], payload["vector_z_m"] = [round(v, 6) for v in vector_state.vector_xyz]
        if vector_state.distance_m is not None:
            payload["distance_m"] = round(vector_state.distance_m, 6)
        return payload

    def _draw_detections(self, image: np.ndarray, detections: list[Detection], color: tuple[int, int, int]) -> np.ndarray:
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return canvas

    def run(self) -> None:
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.grabber.start()
        self.live_preview.start()
        frame_counter = 0
        fps = 0.0
        last_fps_time = time.perf_counter()
        fallback_interval = max(1, int(self.config.runtime.yolo_infer_interval))
        hand_interval = max(1, int(getattr(self.config.runtime, "hand_infer_interval", fallback_interval)))
        target_interval = max(1, int(getattr(self.config.runtime, "target_infer_interval", fallback_interval)))
        if self.config.display.show_preview:
            cv2.namedWindow(self.config.display.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                if self._interrupted:
                    break
                packet = self.grabber.read(self.config.runtime.capture_timeout_ms / 1000.0)
                if packet is None:
                    if self._interrupted:
                        break
                    continue
                self._reload_target_class(packet.index)

                loop_start = time.perf_counter()
                left_bgr, right_bgr = split_stitched_frame(packet.frame, self.config.capture)
                left_bgr, right_bgr = self.rectifier.apply(left_bgr, right_bgr)

                prepared = self.preprocessor.prepare(left_bgr, right_bgr)
                stereo_raw = self.stereo_backend.infer(prepared.left, prepared.right)
                disparity = restore_disparity(stereo_raw, prepared)
                disparity = self._stabilize_disparity(disparity)
                depth_map = disparity_to_depth(disparity, self.config.depth)

                if packet.index == 0 or packet.index % hand_interval == 0:
                    self._cached_hand_detections = self.hand_backend.infer(left_bgr)
                if packet.index == 0 or packet.index % target_interval == 0:
                    self._cached_target_detections = self.target_backend.infer(
                        left_bgr,
                        class_name_filter=self._active_target_class_name,
                    )
                hand_detections = self._cached_hand_detections
                target_detections = self._cached_target_detections
                self.live_preview.update_detections(hand_detections, target_detections)
                vector_state = self._compute_vector_state(left_bgr, depth_map, hand_detections, target_detections)
                payload = self._build_payload(packet.index, vector_state)
                self._append_recent(payload)
                self.writer.write(payload)

                frame_counter += 1
                now = time.perf_counter()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    last_fps_time = now

                if (
                    self.config.runtime.print_every_n_frames > 0
                    and packet.index % self.config.runtime.print_every_n_frames == 0
                ):
                    recent = list(self.recent_payloads)
                    print(
                        json.dumps(
                            {
                                "frame": packet.index,
                                "fps": round(fps, 2),
                                "status": payload["status"],
                                "vector_active": payload["vector_active"],
                                "distance_m": payload["distance_m"],
                                "target_class_name": self._active_target_class_name,
                                "recent_window_size": len(recent),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                annotated_left = self._draw_detections(left_bgr, target_detections, (0, 255, 255))
                annotated_left = self._draw_detections(annotated_left, hand_detections, (0, 255, 0))

                if self.config.display.show_preview:
                    stable_max_disp = self.config.model.max_disp * (
                        float(prepared.source_width) / max(float(prepared.content_width), 1.0)
                    )
                    disparity_color = colorize_disparity(
                        disparity,
                        self.config.display,
                        self.config.depth.min_valid_disp,
                        stable_max_disp,
                    )
                    depth_color = colorize_depth(depth_map, self.config.display, self.config.depth)
                    total_ms = (time.perf_counter() - loop_start) * 1000.0
                    stats = {
                        "fps": fps,
                        "infer_ms": 0.0,
                        "total_ms": total_ms,
                        "frame_index": float(packet.index),
                        "center_depth_m": float(depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2]),
                        "depth_enabled": 1.0,
                    }
                    preview = build_preview(annotated_left, disparity_color, depth_color, stats, self.config.display)
                    cv2.imshow(self.config.display.window_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        except RuntimeError as exc:
            if self._interrupted and "returned no outputs" in str(exc):
                pass
            else:
                raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.live_preview.stop()
            self.grabber.stop()
            self.writer.close()
            cv2.destroyAllWindows()
