from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OcrChunk:
    text: str
    score: float


@dataclass
class OcrResult:
    text: str
    confidence: float
    chunks: list[OcrChunk]


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> OcrResult:
        ...
