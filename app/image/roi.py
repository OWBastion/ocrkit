from __future__ import annotations

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig


def normalize_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def crop_roi(image: np.ndarray, box: RoiBox) -> np.ndarray:
    return image[box.y1:box.y2, box.x1:box.x2].copy()


def crop_all_rois(image: np.ndarray, config: RoiConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    normalized = normalize_canvas(image, config.width, config.height)
    return normalized, {name: crop_roi(normalized, box) for name, box in config.rois.items()}
