from __future__ import annotations

from statistics import mean
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR

from .engine import OcrChunk, OcrResult


class RapidOcrEngine:
    def __init__(self, config_path: Path | None = None) -> None:
        params = None
        if config_path:
            model_dir = config_path.parent
            params = {
                "Det.model_path": str(model_dir / "det.onnx"),
                "Rec.model_path": str(model_dir / "rec.onnx"),
                "Rec.rec_keys_path": str(model_dir / "rec_dict.txt"),
                "Global.use_cls": False,
            }
        self._engine = RapidOCR(config_path=str(config_path) if config_path else None, params=params)

    def recognize(self, image: np.ndarray) -> OcrResult:
        raw = self._engine(image, use_cls=False)
        if not raw.txts or not raw.scores:
            return OcrResult(text="", confidence=0.0, chunks=[])

        chunks = []
        for text, score in zip(raw.txts, raw.scores, strict=True):
            txt = str(text).strip()
            if not txt:
                continue
            chunks.append(OcrChunk(text=txt, score=float(score)))

        if not chunks:
            return OcrResult(text="", confidence=0.0, chunks=[])

        return OcrResult(
            text=" ".join(chunk.text for chunk in chunks),
            confidence=round(float(mean(chunk.score for chunk in chunks)), 4),
            chunks=chunks,
        )
