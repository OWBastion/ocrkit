from __future__ import annotations

import cv2
import numpy as np


def _upscale(image: np.ndarray, ratio: float) -> np.ndarray:
    return cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def preprocess_center_banner(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.GaussianBlur(eq, (3, 3), 0)


def preprocess_left_panel(image: np.ndarray) -> np.ndarray:
    up = _upscale(image, 2.0)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_achievement_panel(image: np.ndarray) -> np.ndarray:
    up = _upscale(image, 3.0)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_run_code_panel(image: np.ndarray) -> np.ndarray:
    return preprocess_left_panel(image)


def preprocess_right_panel(image: np.ndarray) -> np.ndarray:
    up = _upscale(image, 2.0)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(gray, alpha=1.2, beta=8)


def preprocess_by_roi(roi_name: str, image: np.ndarray) -> np.ndarray:
    if roi_name == "center_banner":
        return preprocess_center_banner(image)
    if roi_name == "left_panel":
        return preprocess_left_panel(image)
    if roi_name == "achievement_panel":
        return preprocess_achievement_panel(image)
    if roi_name in {"run_code_panel", "run_code_right_panel"}:
        return preprocess_run_code_panel(image)
    if roi_name == "bottom_left_hero":
        return preprocess_left_panel(image)
    if roi_name == "right_panel":
        return preprocess_right_panel(image)
    return image
