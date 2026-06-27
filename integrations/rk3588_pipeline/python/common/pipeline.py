from __future__ import annotations

import time
from typing import Protocol

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.camera import LatestFrameGrabber, split_stitched_frame
from common.config import AppConfig
from common.preprocess import StereoPreprocessor, restore_disparity
from common.rectify import StereoRectifier
from common.visualize import build_preview, colorize_depth, colorize_disparity, disparity_to_depth


class StereoBackend(Protocol):
    def infer(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        ...


class StereoRuntimePipeline:
    def __init__(self, config: AppConfig, backend: StereoBackend) -> None:
        self.config = config
        self.backend = backend
        self.preprocessor = StereoPreprocessor(config.model)
        self.rectifier = StereoRectifier.from_config(config.rectify)
        self.grabber = LatestFrameGrabber(config.capture)
        self._smoothed_disparity: np.ndarray | None = None

    def _stabilize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        ksize = int(self.config.runtime.median_blur_ksize)
        if ksize > 1:
            if ksize % 2 == 0:
                ksize += 1
            disparity = cv2.medianBlur(disparity.astype(np.float32), ksize)

        alpha = float(self.config.runtime.temporal_smoothing_alpha)
        if alpha <= 0.0:
            return disparity
        alpha = min(alpha, 1.0)

        if self._smoothed_disparity is None or self._smoothed_disparity.shape != disparity.shape:
            self._smoothed_disparity = disparity.copy()
            return disparity

        valid = disparity > self.config.depth.min_valid_disp
        self._smoothed_disparity[valid] = (
            alpha * disparity[valid] + (1.0 - alpha) * self._smoothed_disparity[valid]
        )
        self._smoothed_disparity[~valid] *= 1.0 - alpha
        return self._smoothed_disparity.copy()

    def run(self) -> None:
        self.grabber.start()
        frame_counter = 0
        fps = 0.0
        last_fps_time = time.perf_counter()
        depth_view_enabled = self.config.depth.enabled

        if self.config.display.show_preview:
            cv2.namedWindow(self.config.display.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                packet = self.grabber.read(self.config.runtime.capture_timeout_ms / 1000.0)
                if packet is None:
                    continue

                loop_start = time.perf_counter()
                left_bgr, right_bgr = split_stitched_frame(packet.frame, self.config.capture)
                left_bgr, right_bgr = self.rectifier.apply(left_bgr, right_bgr)

                prepared = self.preprocessor.prepare(left_bgr, right_bgr)
                infer_start = time.perf_counter()
                raw_output = self.backend.infer(prepared.left, prepared.right)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0

                disparity = restore_disparity(raw_output, prepared)
                disparity = self._stabilize_disparity(disparity)
                stable_max_disp = self.config.model.max_disp * (
                    float(prepared.source_width) / max(float(prepared.content_width), 1.0)
                )
                disparity_color = colorize_disparity(
                    disparity,
                    self.config.display,
                    self.config.depth.min_valid_disp,
                    stable_max_disp,
                )

                depth_color = None
                center_depth_m = float("nan")
                if depth_view_enabled:
                    depth = disparity_to_depth(disparity, self.config.depth)
                    depth_color = colorize_depth(depth, self.config.display, self.config.depth)
                    center_depth_m = float(depth[depth.shape[0] // 2, depth.shape[1] // 2])

                frame_counter += 1
                now = time.perf_counter()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    last_fps_time = now

                total_ms = (time.perf_counter() - loop_start) * 1000.0
                stats = {
                    "fps": fps,
                    "infer_ms": infer_ms,
                    "total_ms": total_ms,
                    "frame_index": float(packet.index),
                    "center_depth_m": center_depth_m,
                    "depth_enabled": 1.0 if depth_view_enabled else 0.0,
                }

                if self.config.display.show_preview:
                    preview = build_preview(left_bgr, disparity_color, depth_color, stats, self.config.display)
                    cv2.imshow(self.config.display.window_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("d"):
                        depth_view_enabled = not depth_view_enabled

                if (
                    self.config.runtime.print_every_n_frames > 0
                    and packet.index % self.config.runtime.print_every_n_frames == 0
                ):
                    print(
                        f"frame={packet.index} fps={fps:.2f} infer_ms={infer_ms:.2f} total_ms={total_ms:.2f}",
                        flush=True,
                    )
        finally:
            self.grabber.stop()
            cv2.destroyAllWindows()
