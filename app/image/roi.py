from __future__ import annotations

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
from app.image.quality import is_ultrawide


def normalize_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def select_roi_config(image: np.ndarray, configs: tuple[RoiConfig, ...]) -> RoiConfig:
    aspect_ratio = image.shape[1] / image.shape[0]
    return min(
        configs,
        key=lambda config: abs(aspect_ratio - config.width / config.height),
    )


def crop_roi(image: np.ndarray, box: RoiBox) -> np.ndarray:
    return image[box.y1:box.y2, box.x1:box.x2].copy()


def anchored_rois(config: RoiConfig, canvas_width: int) -> dict[str, RoiBox]:
    """Place ROIs on a canvas wider than the standard one.

    The HUD keeps its height scale and stays attached to the screen edges, so
    left ROIs stay put, right ROIs follow the right edge and central ROIs the middle.
    """
    extra = canvas_width - config.width
    shifted: dict[str, RoiBox] = {}
    for name, box in config.rois.items():
        centre = (box.x1 + box.x2) / 2
        shift = 0 if centre < config.width / 3 else extra if centre > config.width * 2 / 3 else extra // 2
        shifted[name] = RoiBox(box.x1 + shift, box.y1, box.x2 + shift, box.y2)
    return shifted


def crop_all_rois(image: np.ndarray, config: RoiConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    height, width = image.shape[:2]
    if is_ultrawide(width, height):
        normalized = normalize_canvas(image, (config.height * width * 2 + height) // (2 * height), config.height)
    else:
        normalized = normalize_canvas(image, config.width, config.height)
    rois = anchored_rois(config, normalized.shape[1])
    return normalized, {name: crop_roi(normalized, box) for name, box in rois.items()}
