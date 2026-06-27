from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CaptureConfig:
    device_index: int = 0
    device_path: str = ""
    api: str = "auto"
    pixel_format: str = "MJPG"
    buffer_size: int = 1
    frame_width: int = 2560
    frame_height: int = 720
    fps: int = 30
    layout: str = "left_right"
    split_overlap_px: int = 0


@dataclass
class ModelConfig:
    path: str = ""
    input_width: int = 640
    input_height: int = 320
    input_names: list[str] = field(default_factory=lambda: ["left_img", "right_img"])
    output_name: str = "disp_pred"
    input_format: str = "nchw"
    preprocess_mode: str = "resize"
    pad_value: float = 0.0
    max_disp: int = 192
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    cuda_device_id: int = 0
    num_threads: int = 0
    npu_core_mask: str = "auto"


@dataclass
class RectifyConfig:
    enabled: bool = False
    calibration_path: str = ""


@dataclass
class DepthConfig:
    enabled: bool = True
    baseline_m: float = 0.12
    focal_px: float = 905.0
    max_depth_m: float = 30.0
    min_valid_disp: float = 0.1


@dataclass
class DisplayConfig:
    window_name: str = "LightStereo"
    show_preview: bool = True
    show_depth: bool = True
    colormap: str = "turbo"
    preview_scale: float = 1.0
    fixed_disparity_max: float = 0.0
    depth_color_min_m: float = 0.3
    depth_color_max_m: float = 5.0


@dataclass
class RuntimeConfig:
    capture_timeout_ms: int = 1000
    warmup_frames: int = 10
    print_every_n_frames: int = 30
    temporal_smoothing_alpha: float = 0.35
    median_blur_ksize: int = 5
    yolo_infer_interval: int = 1
    hand_infer_interval: int = 1
    target_infer_interval: int = 1


@dataclass
class YoloModelConfig:
    path: str = ""
    task: str = "detect"
    class_names: list[str] = field(default_factory=list)
    input_width: int = 640
    input_height: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    npu_core_mask: str = "auto"


@dataclass
class VectorConfig:
    enabled: bool = True
    hand_class_name: str = "hand"
    target_class_name: str = "cup"
    target_input_path: str = ""
    target_input_format: str = "json"
    target_input_key: str = "target_class_name"
    target_reload_interval_frames: int = 10
    sample_region_px: int = 7
    smoothing_alpha: float = 0.35
    stop_distance_m: float = 0.05
    max_history_seconds: float = 3.0
    output_jsonl_path: str = ""
    output_csv_path: str = ""
    rolling_output_jsonl_path: str = ""
    preview_image_path: str = ""


@dataclass
class ObstacleOutputConfig:
    output_jsonl_path: str = ""
    output_csv_path: str = ""
    rolling_output_jsonl_path: str = ""
    preview_image_path: str = ""


@dataclass
class AppConfig:
    capture: CaptureConfig
    model: ModelConfig
    rectify: RectifyConfig
    depth: DepthConfig
    display: DisplayConfig
    runtime: RuntimeConfig
    hand_model: YoloModelConfig | None = None
    target_model: YoloModelConfig | None = None
    vector: VectorConfig | None = None
    obstacle_output: ObstacleOutputConfig | None = None
    config_path: str = ""


def _resolve_path(base_dir: Path, value: str) -> str:
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _load_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_app_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    base_dir = config_file.parent
    capture = CaptureConfig(**_load_section(raw, "capture"))
    model_raw = _load_section(raw, "model")
    model_raw["path"] = _resolve_path(base_dir, model_raw.get("path", ""))
    model = ModelConfig(**model_raw)

    rectify_raw = _load_section(raw, "rectify")
    rectify_raw["calibration_path"] = _resolve_path(base_dir, rectify_raw.get("calibration_path", ""))
    rectify = RectifyConfig(**rectify_raw)

    depth = DepthConfig(**_load_section(raw, "depth"))
    display = DisplayConfig(**_load_section(raw, "display"))
    runtime = RuntimeConfig(**_load_section(raw, "runtime"))
    hand_model_raw = _load_section(raw, "hand_model")
    if hand_model_raw:
        hand_model_raw["path"] = _resolve_path(base_dir, hand_model_raw.get("path", ""))
        hand_model = YoloModelConfig(**hand_model_raw)
    else:
        hand_model = None

    target_model_raw = _load_section(raw, "target_model")
    if target_model_raw:
        target_model_raw["path"] = _resolve_path(base_dir, target_model_raw.get("path", ""))
        target_model = YoloModelConfig(**target_model_raw)
    else:
        target_model = None

    vector_raw = _load_section(raw, "vector")
    if vector_raw:
        vector_raw["output_jsonl_path"] = _resolve_path(base_dir, vector_raw.get("output_jsonl_path", ""))
        vector_raw["output_csv_path"] = _resolve_path(base_dir, vector_raw.get("output_csv_path", ""))
        vector_raw["target_input_path"] = _resolve_path(base_dir, vector_raw.get("target_input_path", ""))
        vector_raw["rolling_output_jsonl_path"] = _resolve_path(
            base_dir,
            vector_raw.get("rolling_output_jsonl_path", ""),
        )
        vector_raw["preview_image_path"] = _resolve_path(
            base_dir,
            vector_raw.get("preview_image_path", ""),
        )
        vector = VectorConfig(**vector_raw)
    else:
        vector = None

    obstacle_output_raw = _load_section(raw, "obstacle_output")
    if obstacle_output_raw:
        obstacle_output_raw["output_jsonl_path"] = _resolve_path(
            base_dir,
            obstacle_output_raw.get("output_jsonl_path", ""),
        )
        obstacle_output_raw["output_csv_path"] = _resolve_path(
            base_dir,
            obstacle_output_raw.get("output_csv_path", ""),
        )
        obstacle_output_raw["rolling_output_jsonl_path"] = _resolve_path(
            base_dir,
            obstacle_output_raw.get("rolling_output_jsonl_path", ""),
        )
        obstacle_output_raw["preview_image_path"] = _resolve_path(
            base_dir,
            obstacle_output_raw.get("preview_image_path", ""),
        )
        obstacle_output = ObstacleOutputConfig(**obstacle_output_raw)
    else:
        obstacle_output = None

    return AppConfig(
        capture=capture,
        model=model,
        rectify=rectify,
        depth=depth,
        display=display,
        runtime=runtime,
        hand_model=hand_model,
        target_model=target_model,
        vector=vector,
        obstacle_output=obstacle_output,
        config_path=str(config_file),
    )
