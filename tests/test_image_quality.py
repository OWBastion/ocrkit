import cv2
import numpy as np

from app.image.quality import assess_input_quality


def test_quality_preserves_original_size_and_reports_supported_layout() -> None:
    image = np.zeros((1440, 2560, 3), dtype=np.uint8)

    quality = assess_input_quality(image, 1280, 720)

    assert quality["original_size"] == (2560, 1440)
    assert quality["aspect_ratio"] == 1.7778
    assert quality["layout_confidence"] == 1.0
    assert quality["cropped"] is False
    assert quality["warnings"] == []


def test_quality_reports_aspect_ratio_risk_before_normalization() -> None:
    image = np.zeros((1000, 1500, 3), dtype=np.uint8)

    quality = assess_input_quality(image, 1280, 720)

    assert quality["original_size"] == (1500, 1000)
    assert quality["cropped"] is True
    assert quality["layout_confidence"] == 0.0
    assert quality["warnings"] == [
        "quality.aspect_ratio_mismatch",
        "quality.possible_crop",
    ]


def test_blur_score_is_higher_for_flat_or_blurred_input() -> None:
    sharp = np.zeros((720, 1280, 3), dtype=np.uint8)
    sharp[:, ::2] = 255
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)

    sharp_quality = assess_input_quality(sharp, 1280, 720)
    blurred_quality = assess_input_quality(blurred, 1280, 720)

    assert blurred_quality["blur_score"] > sharp_quality["blur_score"]
