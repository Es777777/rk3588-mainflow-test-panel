from __future__ import annotations

from dataclasses import dataclass

import cv2  # type: ignore[import-not-found]
import numpy as np
from rknnlite.api import RKNNLite  # type: ignore[import-not-found]

from common.config import YoloModelConfig


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, max(width - 1, 0))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, max(height - 1, 0))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, max(width - 1, 0))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, max(height - 1, 0))
    return boxes


class RknnLiteYoloBackend:
    def __init__(self, config: YoloModelConfig) -> None:
        self.config = config
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(config.path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {config.path}")

        core_mask_name = config.npu_core_mask.upper()
        core_mask = getattr(RKNNLite, f"NPU_CORE_{core_mask_name}", getattr(RKNNLite, "NPU_CORE_AUTO", 0))
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {config.path}")

    def infer(self, image_bgr: np.ndarray, class_name_filter: str | None = None) -> list[Detection]:
        image, scale, pad_x, pad_y = self._letterbox(image_bgr)
        outputs = self.rknn.inference(inputs=[image])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError(f"RKNN YOLO inference returned no outputs: {self.config.path}")
        detections = self._decode_outputs(
            outputs,
            image_bgr.shape[1],
            image_bgr.shape[0],
            scale,
            pad_x,
            pad_y,
            class_name_filter=class_name_filter,
        )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.config.max_detections]

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        src_h, src_w = image_bgr.shape[:2]
        scale = min(self.config.input_width / src_w, self.config.input_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.config.input_height, self.config.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.config.input_width - new_w) / 2.0
        pad_y = (self.config.input_height - new_h) / 2.0
        x0 = int(round(pad_x))
        y0 = int(round(pad_y))
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb[None, ...], scale, pad_x, pad_y

    def _decode_outputs(
        self,
        outputs: list[np.ndarray],
        src_w: int,
        src_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
        class_name_filter: str | None = None,
    ) -> list[Detection]:
        expected_attrs = 4 + len(self.config.class_names)
        candidate = None
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                continue

            # Prefer layouts that match [num_preds, 4 + num_classes] exactly.
            if expected_attrs > 4:
                if arr.shape[1] == expected_attrs:
                    candidate = arr
                    break
                if arr.shape[0] == expected_attrs:
                    candidate = arr.T
                    break

            if arr.shape[1] >= 6 and arr.shape[0] > arr.shape[1]:
                candidate = arr
                break
            if arr.shape[0] >= 6 and arr.shape[1] > arr.shape[0]:
                candidate = arr.T
                break
        if candidate is None:
            raise RuntimeError("Unsupported YOLO RKNN output layout; expected Nx(4+C) tensor")

        raw = candidate.astype(np.float32)
        boxes = raw[:, :4]
        scores = raw[:, 4:]
        if scores.shape[1] == 0:
            return []

        if np.max(scores) > 1.0 or np.min(scores) < 0.0:
            scores = _sigmoid(scores)

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= float(self.config.conf_threshold)
        if class_name_filter:
            try:
                filter_id = self.config.class_names.index(class_name_filter)
                keep &= class_ids == filter_id
            except ValueError:
                return []
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        boxes = _xywh_to_xyxy(boxes)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-6)
        boxes = _clip_boxes(boxes, src_w, src_h)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confidences.tolist(),
            score_threshold=float(self.config.conf_threshold),
            nms_threshold=float(self.config.iou_threshold),
        )
        if len(indices) == 0:
            return []

        flat_indices = [int(idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx) for idx in indices]
        detections: list[Detection] = []
        for idx in flat_indices:
            class_id = int(class_ids[idx])
            class_name = (
                self.config.class_names[class_id]
                if 0 <= class_id < len(self.config.class_names)
                else str(class_id)
            )
            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[idx]),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                )
            )
        return detections
