from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from training.scripts import evaluate_rec_holdout


def test_holdout_evaluation_writes_a_promotion_gate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crop = tmp_path / "images/holdout/a.png"
    crop.parent.mkdir(parents=True)
    assert cv2.imwrite(str(crop), np.zeros((4, 4, 3), dtype=np.uint8))
    labels = tmp_path / "holdout.txt"
    labels.write_text("images/holdout/a.png\t地图\n", encoding="utf-8")

    class FakeEngine:
        def __init__(self, _config: Path) -> None:
            pass

        def recognize(self, _image: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(text="地图", confidence=0.99)

    monkeypatch.setattr(evaluate_rec_holdout, "RapidOcrEngine", FakeEngine)
    report = evaluate_rec_holdout.evaluate(labels, tmp_path, tmp_path / "rapidocr.yaml", 1.0)

    assert report["status"] == "passed"
    assert report["matched"] == 1
    assert report["total"] == 1


def test_holdout_evaluation_rejects_crops_outside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = tmp_path / "holdout.txt"
    labels.write_text("../outside.png\t文本\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_rec_holdout, "RapidOcrEngine", lambda _config: object())

    with pytest.raises(RuntimeError, match="outside the images root"):
        evaluate_rec_holdout.evaluate(labels, tmp_path, tmp_path / "rapidocr.yaml", 1.0)
