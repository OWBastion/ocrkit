from __future__ import annotations

import sys

import pytest

import scripts.batch_eval as batch_eval


def test_main_rejects_fixture_accuracy_below_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch_eval,
        "evaluate",
        lambda *_args: {"field_accuracy": 0.965, "matched_fields": 366, "total_fields": 379},
    )
    monkeypatch.setattr(sys, "argv", ["batch_eval.py", "--min-field-accuracy", "0.9657"])

    with pytest.raises(SystemExit, match="fixture field accuracy"):
        batch_eval.main()


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
