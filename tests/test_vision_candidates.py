from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
from training.scripts.prepare_rec_candidates import prepare_candidates
from training.vision import VisionLine


BOX = np.array([[[2, 3], [12, 3], [12, 9], [2, 9]]], dtype=np.float32)


def _fixtures(tmp_path: Path) -> tuple[Path, RoiConfig]:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1]
    (fixtures / "sample.png").write_bytes(encoded.tobytes())
    (fixtures / "cases.json").write_text(json.dumps([{"id": "sample_01", "image": "sample.png"}]), encoding="utf-8")
    return fixtures, RoiConfig(width=60, height=40, rois={"left_panel": RoiBox(0, 0, 30, 20)})


def _rapid_factory(text: str, confidence: float) -> type[object]:
    class FakeRapidOcr:
        def __call__(self, image: np.ndarray, use_cls: bool) -> SimpleNamespace:
            assert use_cls is False
            return SimpleNamespace(boxes=BOX, txts=(text,), scores=(confidence,))

    return FakeRapidOcr


def _vision_factory(text: str, confidence: float) -> type[object]:
    class FakeVisionOcr:
        def recognize(self, image: np.ndarray) -> list[VisionLine]:
            return [VisionLine(text, confidence, BOX[0])]

    return FakeVisionOcr


def _review_row(tmp_path: Path, rapid_confidence: float, vision_confidence: float) -> tuple[dict[str, object], dict[str, int]]:
    fixtures, config = _fixtures(tmp_path)
    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=_rapid_factory("Ａ　挑战", rapid_confidence),
        vision_factory=_vision_factory("A 挑战", vision_confidence),
    )
    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    return row, summary


def test_matching_high_confidence_vision_and_rapidocr_candidates_are_auto_accepted(tmp_path: Path) -> None:
    row, summary = _review_row(tmp_path, 0.98, 0.99)

    assert summary["auto_accepted"] == 1
    assert row["review_status"] == "accepted"
    assert row["transcription"] == "A 挑战"
    assert row["auto_accept_reason"] == "rapidocr_vision_agreement"
    assert row["rapidocr_text"] == "Ａ　挑战"
    assert row["vision_text"] == "A 挑战"


def test_low_confidence_matching_candidates_remain_pending(tmp_path: Path) -> None:
    row, summary = _review_row(tmp_path, 0.9799, 0.99)

    assert summary["auto_accepted"] == 0
    assert row["review_status"] == "pending"
    assert row["transcription"] is None
    assert row["auto_accept_reason"] is None


def test_teacher_model_adds_train_suggestion_without_overwriting_review_status(tmp_path: Path) -> None:
    fixtures, config = _fixtures(tmp_path)

    class EmptyVision:
        def recognize(self, image: np.ndarray) -> list[VisionLine]:
            return []

    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=_rapid_factory("A 挑战", 0.99),
        vision_factory=EmptyVision,
        teacher_model_version="2026.07.31-110827",
        teacher_ocr_factory=_rapid_factory("A 挑战", 0.99),
    )

    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert summary["teacher_model_version"] == "2026.07.31-110827"
    assert summary["teacher_suggestions"] == 1
    assert summary["teacher_auto_accept_eligible"] == 1
    assert row["review_status"] == "pending"
    assert row["teacher_text"] == "A 挑战"
    assert row["suggested_transcription"] == "A 挑战"
