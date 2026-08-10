from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.batch_eval as batch_eval


def test_evaluate_uses_requested_model_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([{"id": "case", "image": "fixture.png", "expected": {"map_name": "Hanamura"}}]),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    context = SimpleNamespace(
        roi_config="roi",
        map_names="maps",
        map_aliases="aliases",
        ocr_engine="builtin",
        engine_name="rapidocr",
        model_version="builtin",
        layout_version="1280x720-v3",
    )

    class StubEngine:
        def __init__(self, config_path: Path) -> None:
            captured["config_path"] = config_path

    monkeypatch.setattr(batch_eval, "create_context", lambda: context)
    monkeypatch.setattr(batch_eval, "RapidOcrEngine", StubEngine)
    monkeypatch.setattr(batch_eval.cv2, "imread", lambda _: np.zeros((1, 1, 3), dtype=np.uint8))
    def extract_stub(*_args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "include_debug": False,
            "request_id": "fixture:case",
            "engine_name": "rapidocr",
            "model_version": "builtin",
            "layout_version": "1280x720-v3",
        }
        return SimpleNamespace(data=SimpleNamespace(model_dump=lambda: {"map_name": "Hanamura"}))

    monkeypatch.setattr(batch_eval, "extract_structured", extract_stub)
    model_config = tmp_path / "rapidocr.yaml"

    result = batch_eval.evaluate(cases_path, tmp_path, model_config)

    assert captured["config_path"] == model_config
    assert context.ocr_engine.__class__ is StubEngine
    assert result["matched_fields"] == 1
    assert result["total_fields"] == 1
    assert result["field_accuracy"] == 1.0


def test_main_rejects_fixture_accuracy_below_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch_eval,
        "evaluate",
        lambda *_args: {"field_accuracy": 0.965, "matched_fields": 366, "total_fields": 379},
    )
    monkeypatch.setattr(sys, "argv", ["batch_eval.py", "--min-field-accuracy", "0.9657"])

    with pytest.raises(SystemExit, match="fixture field accuracy"):
        batch_eval.main()


def test_main_writes_report_before_rejecting_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = tmp_path / "fixture_report.json"
    monkeypatch.setattr(
        batch_eval,
        "evaluate",
        lambda *_args: {"field_accuracy": 0.9, "matched_fields": 9, "total_fields": 10},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_eval.py", "--report", str(report), "--min-field-accuracy", "0.9657"],
    )

    with pytest.raises(SystemExit, match="fixture field accuracy"):
        batch_eval.main()

    assert json.loads(report.read_text(encoding="utf-8"))["matched_fields"] == 9


def test_main_rejects_run_code_fixture_accuracy_below_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter(
        [
            {"field_accuracy": 1.0, "matched_fields": 10, "total_fields": 10},
            {"field_accuracy": 0.5, "matched_fields": 1, "total_fields": 2},
        ]
    )
    monkeypatch.setattr(batch_eval, "evaluate", lambda *_args: next(results))
    monkeypatch.setattr(sys, "argv", ["batch_eval.py", "--min-run-code-accuracy", "1.0"])

    with pytest.raises(SystemExit, match="run-code fixture exact-match accuracy"):
        batch_eval.main()
