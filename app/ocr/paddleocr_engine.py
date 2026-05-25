from __future__ import annotations

from statistics import mean

import numpy as np

from .engine import OcrChunk, OcrResult


class PaddleOcrEngine:
    def __init__(self, lang: str = "ch") -> None:
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False, use_gpu=False)

    def recognize(self, image: np.ndarray) -> OcrResult:
        raw = self._engine.ocr(image, cls=False)
        if not raw:
            return OcrResult(text="", confidence=0.0, chunks=[])

        chunks: list[OcrChunk] = []
        for block in raw:
            for item in block:
                if not item or len(item) < 2:
                    continue
                txt = str(item[1][0]).strip()
                if not txt:
                    continue
                score = float(item[1][1])
                chunks.append(OcrChunk(text=txt, score=score))

        if not chunks:
            return OcrResult(text="", confidence=0.0, chunks=[])

        return OcrResult(
            text=" ".join(chunk.text for chunk in chunks),
            confidence=round(float(mean(chunk.score for chunk in chunks)), 4),
            chunks=chunks,
        )
