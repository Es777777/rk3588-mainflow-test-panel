from __future__ import annotations

import math
from functools import lru_cache

import cv2  # type: ignore[import-not-found]
import numpy as np

from common.config import DepthConfig, DisplayConfig


_COLORMAPS = {
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "jet": cv2.COLORMAP_JET,
    "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
    "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
}


@lru_cache(maxsize=1)
def _get_screen_size() -> tuple[int, int] | None:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None
    return None


def fit_preview_to_screen(image: np.ndarray, display: DisplayConfig) -> np.ndarray:
    scale = float(display.preview_scale)
    if scale <= 0.0:
        scale = 1.0

    screen_size = _get_screen_size()
    if screen_size is not None:
        screen_width, screen_height = screen_size
        max_width = max(640, int(screen_width * 0.95))
        max_height = max(360, int(screen_height * 0.9))
        fit_scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
        scale *= fit_scale

    if abs(scale - 1.0) < 1e-6:
        return image

    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def disparity_to_depth(disparity: np.ndarray, config: DepthConfig) -> np.ndarray:
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > config.min_valid_disp
    depth[valid] = (config.focal_px * config.baseline_m) / disparity[valid]
    depth = np.clip(depth, 0.0, config.max_depth_m)
    return depth


def colorize_disparity(
    disparity: np.ndarray,
    display: DisplayConfig,
    min_valid_disp: float,
    max_disp: float | None = None,
) -> np.ndarray:
    valid = disparity > min_valid_disp
    if not np.any(valid):
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    max_value = float(max_disp or display.fixed_disparity_max)
    if max_value <= min_valid_disp:
        max_value = float(np.percentile(disparity[valid], 98.0))
    max_value = max(max_value, min_valid_disp + 1e-3)
    scaled = np.clip(disparity / max_value, 0.0, 1.0)
    image = (scaled * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image, _COLORMAPS.get(display.colormap.lower(), cv2.COLORMAP_JET))
    color[~valid] = 0
    return color


def colorize_depth(depth: np.ndarray, display: DisplayConfig, config: DepthConfig) -> np.ndarray:
    valid = depth > 0.0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    min_depth = max(float(display.depth_color_min_m), 0.0)
    max_depth = float(display.depth_color_max_m)
    if max_depth <= min_depth:
        max_depth = max(float(config.max_depth_m), min_depth + 1e-3)

    scaled_depth = (depth - min_depth) / max(max_depth - min_depth, 1e-3)
    scaled = 1.0 - np.clip(scaled_depth, 0.0, 1.0)
    image = (scaled * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image, _COLORMAPS.get(display.colormap.lower(), cv2.COLORMAP_JET))
    color[~valid] = 0
    return color


def overlay_stats(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return canvas


def build_preview(
    left_bgr: np.ndarray,
    disparity_color: np.ndarray,
    depth_color: np.ndarray | None,
    stats: dict[str, float],
    display: DisplayConfig,
) -> np.ndarray:
    tiles = [left_bgr, disparity_color]
    if depth_color is not None and display.show_depth:
        tiles.append(depth_color)
    preview = np.hstack(tiles)

    lines = [
        f"fps: {stats.get('fps', 0.0):5.2f}",
        f"infer: {stats.get('infer_ms', 0.0):6.2f} ms",
        f"total: {stats.get('total_ms', 0.0):6.2f} ms",
        f"frame: {int(stats.get('frame_index', 0))}",
        f"depth: {'on' if bool(stats.get('depth_enabled', 0.0)) else 'off'}",
        "keys: d toggle, q quit",
    ]
    if math.isfinite(stats.get("center_depth_m", float("nan"))):
        lines.append(f"center depth: {stats['center_depth_m']:.2f} m")

    preview = overlay_stats(preview, lines)
    return fit_preview_to_screen(preview, display)
