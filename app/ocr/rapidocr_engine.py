from __future__ import annotations

from statistics import mean

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .engine import OcrChunk, OcrResult


class RapidOcrEngine:
    def __init__(self) -> None:
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray) -> OcrResult:
        raw, _ = self._engine(image)
        if not raw:
            return OcrResult(text="", confidence=0.0, chunks=[])

        chunks: list[OcrChunk] = []
        for item in raw:
            if len(item) < 3:
                continue
            txt = str(item[1]).strip()
            if not txt:
                continue
            score = float(item[2])
            chunks.append(OcrChunk(text=txt, score=score))

        if not chunks:
            return OcrResult(text="", confidence=0.0, chunks=[])

        return OcrResult(
            text=" ".join(chunk.text for chunk in chunks),
            confidence=round(float(mean(chunk.score for chunk in chunks)), 4),
            chunks=chunks,
        )
