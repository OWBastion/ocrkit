from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OcrChunk:
    text: str
    score: float
    # Bounding rectangle (x1, y1, x2, y2) in the coordinates of the image given to the engine, when known.
    box: tuple[float, float, float, float] | None = None


@dataclass
class OcrResult:
    text: str
    confidence: float
    chunks: list[OcrChunk]


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> OcrResult:
        ...
