from __future__ import annotations

from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.config import ModelConfig


@dataclass
class PreparedStereo:
    left: np.ndarray
    right: np.ndarray
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    content_x: int
    content_y: int
    content_width: int
    content_height: int


class StereoPreprocessor:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.mean = np.asarray(config.mean, dtype=np.float32)
        self.std = np.asarray(config.std, dtype=np.float32)

    def prepare(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> PreparedStereo:
        source_height, source_width = left_bgr.shape[:2]
        left, left_meta = self._prepare_single(left_bgr)
        right, right_meta = self._prepare_single(right_bgr)
        if left_meta != right_meta:
            raise ValueError("Left and right preprocessing metadata do not match")

        content_x, content_y, content_width, content_height = left_meta
        return PreparedStereo(
            left=left,
            right=right,
            source_width=source_width,
            source_height=source_height,
            input_width=self.config.input_width,
            input_height=self.config.input_height,
            content_x=content_x,
            content_y=content_y,
            content_width=content_width,
            content_height=content_height,
        )

    def _prepare_single(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        mode = self.config.preprocess_mode.lower()
        if mode == "pad":
            prepared, meta = self._prepare_single_pad(image_bgr)
        elif mode == "resize_right_top_pad":
            prepared, meta = self._prepare_single_resize_right_top_pad(image_bgr)
        else:
            prepared, meta = self._prepare_single_resize(image_bgr)

        if self.config.input_format.lower() == "nhwc":
            return prepared[None, ...].astype(np.float32), meta
        return np.transpose(prepared, (2, 0, 1))[None, ...].astype(np.float32), meta

    def _prepare_single_resize(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        resized = cv2.resize(
            image_bgr,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        return normalized.astype(np.float32), (0, 0, self.config.input_width, self.config.input_height)

    def _prepare_single_pad(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        source_height, source_width = image_bgr.shape[:2]
        if source_width > self.config.input_width or source_height > self.config.input_height:
            raise ValueError(
                "Pad mode requires source image to fit within the configured model input size; "
                f"got source={source_width}x{source_height}, input={self.config.input_width}x{self.config.input_height}"
            )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std

        canvas = np.full(
            (self.config.input_height, self.config.input_width, 3),
            float(self.config.pad_value),
            dtype=np.float32,
        )
        offset_x = (self.config.input_width - source_width) // 2
        offset_y = (self.config.input_height - source_height) // 2
        canvas[offset_y : offset_y + source_height, offset_x : offset_x + source_width, :] = normalized
        return canvas, (offset_x, offset_y, source_width, source_height)

    def _prepare_single_resize_right_top_pad(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        source_height, source_width = image_bgr.shape[:2]
        scale = min(self.config.input_width / source_width, self.config.input_height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))

        resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std

        pad_top = self.config.input_height - resized_height
        pad_right = self.config.input_width - resized_width
        if pad_top < 0 or pad_right < 0:
            raise ValueError(
                "Resize-right-top-pad produced invalid target size; "
                f"resized={resized_width}x{resized_height}, input={self.config.input_width}x{self.config.input_height}"
            )

        padded = np.pad(normalized, ((pad_top, 0), (0, pad_right), (0, 0)), mode="edge")
        return padded.astype(np.float32), (0, pad_top, resized_width, resized_height)


def restore_disparity(raw_output: np.ndarray, prepared: PreparedStereo) -> np.ndarray:
    disparity = np.asarray(raw_output, dtype=np.float32)
    if disparity.ndim == 4:
        if disparity.shape[1] == 1:
            disparity = disparity[0, 0]
        elif disparity.shape[-1] == 1:
            disparity = disparity[0, :, :, 0]
        else:
            disparity = disparity[0]
    elif disparity.ndim == 3:
        if disparity.shape[0] == 1:
            disparity = disparity[0]
        elif disparity.shape[-1] == 1:
            disparity = disparity[:, :, 0]
    disparity = np.maximum(disparity, 0.0)

    input_height, input_width = disparity.shape[:2]
    scale_x = float(input_width) / float(prepared.input_width)
    scale_y = float(input_height) / float(prepared.input_height)

    content_x = int(round(prepared.content_x * scale_x))
    content_y = int(round(prepared.content_y * scale_y))
    content_width = int(round(prepared.content_width * scale_x))
    content_height = int(round(prepared.content_height * scale_y))

    content_x = min(max(content_x, 0), max(0, input_width - 1))
    content_y = min(max(content_y, 0), max(0, input_height - 1))
    content_width = max(1, min(content_width, input_width - content_x))
    content_height = max(1, min(content_height, input_height - content_y))

    disparity = disparity[content_y : content_y + content_height, content_x : content_x + content_width]

    restored = cv2.resize(
        disparity,
        (prepared.source_width, prepared.source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    restored *= float(prepared.source_width) / float(content_width)
    return restored
