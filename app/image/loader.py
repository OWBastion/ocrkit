from __future__ import annotations

import cv2
import numpy as np


SUPPORTED_MIME = {"image/png", "image/jpeg", "image/webp"}


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image")
    return image
