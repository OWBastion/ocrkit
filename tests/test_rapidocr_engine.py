from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import app.ocr.rapidocr_engine as rapidocr_engine


def test_rapidocr_engine_uses_config_and_disables_classifier(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class StubRapidOCR:
        def __init__(self, config_path: str | None = None, params=None) -> None:
            calls["config_path"] = config_path
            calls["params"] = params

        def __call__(self, image: np.ndarray, **kwargs):
            calls["kwargs"] = kwargs
            return SimpleNamespace(txts=("one", "two"), scores=(0.8, 0.6))

    monkeypatch.setattr(rapidocr_engine, "RapidOCR", StubRapidOCR)
    config_path = tmp_path / "rapidocr.yaml"
    engine = rapidocr_engine.RapidOcrEngine(config_path)

    result = engine.recognize(np.zeros((1, 1, 3), dtype=np.uint8))
    assert calls["config_path"] == str(config_path)
    assert calls["params"] == {
        "Det.model_path": str(tmp_path / "det.onnx"),
        "Rec.model_path": str(tmp_path / "rec.onnx"),
        "Rec.rec_keys_path": str(tmp_path / "rec_dict.txt"),
        "Global.use_cls": False,
    }
    assert calls["kwargs"] == {"use_cls": False}
    assert result.text == "one two"
    assert result.confidence == 0.7
