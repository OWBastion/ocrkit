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

        raw_boxes = getattr(raw, "boxes", None)
        boxes = raw_boxes if raw_boxes is not None and len(raw_boxes) == len(raw.txts) else [None] * len(raw.txts)
        chunks = []
        for text, score, box in zip(raw.txts, raw.scores, boxes, strict=True):
            txt = str(text).strip()
            if not txt:
                continue
            rect = None
            if box is not None:
                points = np.asarray(box, dtype=float).reshape(-1, 2)
                rect = (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))
            chunks.append(OcrChunk(text=txt, score=float(score), box=rect))

        if not chunks:
            return OcrResult(text="", confidence=0.0, chunks=[])

        return OcrResult(
            text=" ".join(chunk.text for chunk in chunks),
            confidence=round(float(mean(chunk.score for chunk in chunks)), 4),
            chunks=chunks,
        )
