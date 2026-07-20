from __future__ import annotations

import cv2
import numpy as np


_ASPECT_TOLERANCE = 0.03
_BLUR_VARIANCE_REFERENCE = 1000.0


def _relative_aspect_difference(aspect_ratio: float, target_ratio: float) -> float:
    return abs(aspect_ratio - target_ratio) / target_ratio


def assess_input_quality(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> dict[str, float | bool | str | tuple[int, int] | list[str]]:
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_ratio = target_width / target_height
    aspect_difference = _relative_aspect_difference(aspect_ratio, target_ratio)
    cropped = aspect_difference > _ASPECT_TOLERANCE
    layout_confidence = max(0.0, min(1.0, 1.0 - aspect_difference / _ASPECT_TOLERANCE))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, np.log1p(laplacian_variance) / np.log1p(_BLUR_VARIANCE_REFERENCE))
    blur_score = 1.0 - sharpness

    warnings: list[str] = []
    if cropped:
        warnings.append("quality.aspect_ratio_mismatch")
        warnings.append("quality.possible_crop")

    return {
        "original_size": (width, height),
        "aspect_ratio": round(aspect_ratio, 4),
        "layout_confidence": round(layout_confidence, 4),
        "cropped": cropped,
        "blur_score": round(blur_score, 4),
        "warnings": warnings,
    }
