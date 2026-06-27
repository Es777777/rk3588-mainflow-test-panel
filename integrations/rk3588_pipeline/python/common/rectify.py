from __future__ import annotations

import json
from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.config import RectifyConfig


@dataclass
class StereoRectifier:
    enabled: bool
    left_map_x: np.ndarray | None = None
    left_map_y: np.ndarray | None = None
    right_map_x: np.ndarray | None = None
    right_map_y: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: RectifyConfig) -> "StereoRectifier":
        if not config.enabled:
            return cls(enabled=False)

        with open(config.calibration_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_size = (int(data["image_width"]), int(data["image_height"]))
        k1 = np.asarray(data["left_camera_matrix"], dtype=np.float64)
        d1 = np.asarray(data["left_distortion"], dtype=np.float64)
        k2 = np.asarray(data["right_camera_matrix"], dtype=np.float64)
        d2 = np.asarray(data["right_distortion"], dtype=np.float64)
        r = np.asarray(data["rotation"], dtype=np.float64)
        t = np.asarray(data["translation"], dtype=np.float64).reshape(3, 1)

        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            image_size,
            r,
            t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            k1, d1, r1, p1, image_size, cv2.CV_32FC1
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            k2, d2, r2, p2, image_size, cv2.CV_32FC1
        )

        return cls(
            enabled=True,
            left_map_x=left_map_x,
            left_map_y=left_map_y,
            right_map_x=right_map_x,
            right_map_y=right_map_y,
        )

    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return left, right
        rect_left = cv2.remap(left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        return rect_left, rect_right
