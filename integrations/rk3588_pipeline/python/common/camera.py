from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.config import CaptureConfig


@dataclass
class FramePacket:
    index: int
    timestamp_s: float
    frame: np.ndarray


def _api_to_backend(api: str) -> int:
    value = api.lower()
    mapping = {
        "auto": getattr(cv2, "CAP_ANY", 0),
        "dshow": getattr(cv2, "CAP_DSHOW", getattr(cv2, "CAP_ANY", 0)),
        "msmf": getattr(cv2, "CAP_MSMF", getattr(cv2, "CAP_ANY", 0)),
        "v4l2": getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_ANY", 0)),
    }
    return mapping.get(value, getattr(cv2, "CAP_ANY", 0))


def _pixel_format_to_fourcc(pixel_format: str) -> int | None:
    value = str(pixel_format or "").strip().upper()
    if len(value) == 4:
        return cv2.VideoWriter_fourcc(*value)
    return None


class LatestFrameGrabber:
    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._backend = _api_to_backend(config.api)
        self._primary_source: int | str = config.device_path if config.device_path else config.device_index
        self._fallback_source: int | str = config.device_index
        self.cap = self._open_capture()

        self._running = False
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: FramePacket | None = None
        self._new_frame = False
        self._frame_listeners: list = []

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def add_frame_listener(self, callback) -> None:
        if callback is not None:
            self._frame_listeners.append(callback)

    def read(self, timeout_s: float) -> FramePacket | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._new_frame or not self._running, timeout=timeout_s):
                return None
            if self._latest is None:
                return None
            packet = FramePacket(
                index=self._latest.index,
                timestamp_s=self._latest.timestamp_s,
                frame=self._latest.frame.copy(),
            )
            self._new_frame = False
            return packet

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._primary_source, self._backend)
        self._apply_capture_settings(cap)
        if cap.isOpened():
            return cap

        if self.config.device_path:
            cap.release()
            cap = cv2.VideoCapture(self._fallback_source, self._backend)
            self._apply_capture_settings(cap)
            if cap.isOpened():
                return cap

        raise RuntimeError(
            f"Failed to open capture device {self.config.device_path or self.config.device_index}"
        )

    def _apply_capture_settings(self, cap: cv2.VideoCapture) -> None:
        fourcc = _pixel_format_to_fourcc(getattr(self.config, "pixel_format", "MJPG"))
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(getattr(self.config, "buffer_size", 1))))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def _reopen_capture(self) -> bool:
        try:
            old_cap = self.cap
            self.cap = self._open_capture()
            old_cap.release()
            return True
        except Exception:
            return False

    def _capture_loop(self) -> None:
        frame_index = 0
        consecutive_failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    self._reopen_capture()
                    consecutive_failures = 0
                    time.sleep(0.05)
                    continue
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            packet = FramePacket(index=frame_index, timestamp_s=time.perf_counter(), frame=frame)
            with self._cond:
                self._latest = packet
                self._new_frame = True
                self._cond.notify_all()
            for callback in list(self._frame_listeners):
                try:
                    callback(packet)
                except Exception:
                    pass
            frame_index += 1


def split_stitched_frame(frame: np.ndarray, config: CaptureConfig) -> tuple[np.ndarray, np.ndarray]:
    overlap = max(0, int(config.split_overlap_px))
    if config.layout in {"left_right", "right_left"}:
        width = frame.shape[1]
        center = width // 2
        left_end = min(width, center + overlap // 2)
        right_start = max(0, center - overlap // 2)
        first = frame[:, :left_end].copy()
        second = frame[:, right_start:].copy()
        if config.layout == "right_left":
            return second, first
        return first, second

    if config.layout in {"top_bottom", "bottom_top"}:
        height = frame.shape[0]
        center = height // 2
        top_end = min(height, center + overlap // 2)
        bottom_start = max(0, center - overlap // 2)
        first = frame[:top_end, :].copy()
        second = frame[bottom_start:, :].copy()
        if config.layout == "bottom_top":
            return second, first
        return first, second

    raise ValueError(f"Unsupported capture.layout: {config.layout}")
