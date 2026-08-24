from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.compatibility_gate as compatibility_gate


def test_matrix_records_producer_revision_without_copying_bastion_contract() -> None:
    matrix = compatibility_gate.load_matrix(Path("configs/bastion_screenshot_compatibility.json"))

    assert matrix["producer_contract"] == {
        "repository": "OWBastion/Bastion",
        "revision": "settlement-hud-v1",
        "minimum_released_version": "v26.0811.1",
    }
    assert {item["layout_version"] for item in matrix["supported_layouts"]} == {
        "1280x720-v6",
        "1280x800-v1",
    }
    assert "run_code" in matrix["critical_fields"]


def test_gate_fails_a_critical_field_even_when_other_fields_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer_contract": {"revision": "settlement-hud-v1", "minimum_released_version": "v26.0811.1"},
                "supported_producer_revisions": [{"revision": "settlement-hud-v1"}],
                "supported_layouts": [{"layout_version": "1280x720-v6", "aspect_ratio": "16:9"}],
                "critical_fields": ["run_code", "version"],
                "fixture_sets": [{"id": "safe", "producer_revision": "settlement-hud-v1", "cases": "cases.json", "images": "images", "critical_fields": ["run_code", "version"], "required_layouts": ["1280x720-v6"]}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cases.json").write_text("[]", encoding="utf-8")
    (tmp_path / "images").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        compatibility_gate,
        "evaluate",
        lambda *_args: {
            "field_counts": {"run_code": {"matched": 0, "total": 1}, "version": {"matched": 1, "total": 1}},
            "results": [
                {
                    "id": "run-code-regression",
                    "layout_version": "1280x720-v6",
                    "quality_warnings": [],
                    "fields": {
                        "run_code": {"expected": "4821-7354-1926", "actual": None, "matched": False},
                        "version": {"expected": "26.0822.1", "actual": "26.0822.1", "matched": True},
                    },
                }
            ],
        },
    )

    report = compatibility_gate.run_gate(matrix_path)

    assert report["ok"] is False
    assert report["failures"] == [
        {
            "fixture_set": "safe",
            "case": "run-code-regression",
            "field": "run_code",
            "classification": "recognition/model accuracy regression",
        }
    ]
