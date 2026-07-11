from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionLine:
    text: str
    confidence: float
    box: np.ndarray


class VisionOcr:
    def __init__(self) -> None:
        try:
            import Quartz
            from Foundation import NSData
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Apple Vision requires macOS and the training extra: uv sync --extra vision"
            ) from exc

        self._quartz = Quartz
        self._data_type = NSData
        self._handler_type = VNImageRequestHandler
        self._request_type = VNRecognizeTextRequest
        self._accuracy_level = VNRequestTextRecognitionLevelAccurate

    def recognize(self, image: np.ndarray) -> list[VisionLine]:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("failed to encode image for Apple Vision")

        request = self._request_type.alloc().initWithCompletionHandler_(lambda _request, _error: None)
        request.setRecognitionLevel_(self._accuracy_level)
        request.setRecognitionLanguages_(["zh-Hans", "en-US"])
        request.setAutomaticallyDetectsLanguage_(False)
        request.setUsesLanguageCorrection_(False)

        image_data = self._data_type.dataWithBytes_length_(encoded.tobytes(), len(encoded))
        ci_image = self._quartz.CIImage.imageWithData_(image_data)
        handler = self._handler_type.alloc().initWithCIImage_options_(ci_image, None)
        success, error = handler.performRequests_error_([request], None)
        if not success and error is not None:
            raise RuntimeError(f"Apple Vision recognition failed: {error}")

        height, width = image.shape[:2]
        lines: list[VisionLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string()).strip()
            if not text:
                continue
            bounds = observation.boundingBox()
            if len(bounds) == 2:
                (x, y), (box_width, box_height) = bounds
            else:
                x, y, box_width, box_height = bounds
            x1 = float(x * width)
            x2 = float((x + box_width) * width)
            y1 = float((1 - y - box_height) * height)
            y2 = float((1 - y) * height)
            lines.append(
                VisionLine(
                    text=text,
                    confidence=float(candidate.confidence()),
                    box=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32),
                )
            )
        return lines
