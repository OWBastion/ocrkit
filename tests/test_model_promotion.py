from __future__ import annotations

import hashlib
import json

import pytest

from app.model_artifacts.release import compare_manifests
from training.scripts import promote_model_channel, rollback_model_channel


class _Body:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class _Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.puts: list[dict[str, object]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, _Body]:  # noqa: N803 - boto3 shape
        return {"Body": _Body(self.objects[Key])}

    def put_object(self, **kwargs: object) -> None:
        self.puts.append(kwargs)
        self.objects[str(kwargs["Key"])] = bytes(kwargs["Body"])


def _manifest(version: str, evidence: dict[str, object] | None = None) -> bytes:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "version": version,
        "files": {
            name: {
                "object_key": f"models/pp-ocrv6-small/{version}/{name}",
                "sha256": hashlib.sha256(b"artifact").hexdigest(),
                "size_bytes": 8,
            }
            for name in ("det.onnx", "rec.onnx", "rec_dict.txt", "rapidocr.yaml")
        },
    }
    if evidence is not None:
        payload["release_evidence"] = evidence
    return json.dumps(payload).encode()


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation": {
            "fixture": {
                "status": "passed",
                "field_accuracy": 0.98,
                "run_code": {"field_accuracy": 1.0},
                "field_metrics": {"map_name": {"accuracy": 1.0}},
            },
            "holdout": {"status": "passed", "accuracy": 0.97},
            "comparison": {
                "verdict": "passed",
                "reasons": [],
                "stable": {"version": "stable-v1", "manifest_key": "models/pp-ocrv6-small/stable-v1/manifest.json"},
                "candidate": {
                    "fixture": {"field_accuracy": 0.98, "run_code_accuracy": 1.0, "field_metrics": {"map_name": {"matched": 40, "total": 40}}},
                    "holdout": {"accuracy": 0.97},
                },
                "baseline": {
                    "fixture": {"field_accuracy": 0.97, "run_code_accuracy": 1.0, "field_metrics": {"map_name": {"matched": 39, "total": 40}}},
                    "holdout": {"accuracy": 0.95},
                },
            },
        },
        "full_test_suite": {"status": "passed"},
        "provenance": {"status": "recorded", "source": {"snapshot": "s1@v1"}},
        "compatibility": {"status": "passed", "report": {"schema_version": 1, "ok": True}},
    }


def _objects(stable_history: list[dict[str, object]] | None = None) -> _Client:
    stable = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "manifest_key": "models/pp-ocrv6-small/stable-v1/manifest.json",
    }
    if stable_history is not None:
        stable["history"] = stable_history
    return _Client({
        "models/pp-ocrv6-small/channels/candidate.json": json.dumps({
            "schema_version": 1,
            "model": "pp-ocrv6-small",
            "manifest_key": "models/pp-ocrv6-small/candidate-v2/manifest.json",
        }).encode(),
        "models/pp-ocrv6-small/channels/stable.json": json.dumps(stable).encode(),
        "models/pp-ocrv6-small/candidate-v2/manifest.json": _manifest("candidate-v2", _evidence()),
        "models/pp-ocrv6-small/stable-v1/manifest.json": _manifest("stable-v1"),
    })


def test_compare_fails_closed_when_candidate_evidence_is_incomplete() -> None:
    candidate = json.loads(_manifest("candidate-v2"))
    stable = json.loads(_manifest("stable-v1"))

    report = compare_manifests("models/pp-ocrv6-small/candidate-v2/manifest.json", candidate, "models/pp-ocrv6-small/stable-v1/manifest.json", stable)

    assert report["eligible"] is False
    assert "missing or failing holdout evidence" in report["reasons"]
    assert "missing or failing provenance evidence" in report["reasons"]


def test_promote_requires_evidence_and_does_not_write_stable_on_failure() -> None:
    client = _objects()
    incomplete = json.loads(_manifest("candidate-v2"))
    client.objects["models/pp-ocrv6-small/candidate-v2/manifest.json"] = json.dumps(incomplete).encode()

    with pytest.raises(ValueError, match="not eligible"):
        promote_model_channel.promote(client, "models", "models/pp-ocrv6-small/channels/candidate.json", "models/pp-ocrv6-small/channels/stable.json")

    assert client.puts == []


def test_promote_writes_only_stable_pointer_after_verification(monkeypatch) -> None:
    client = _objects()
    monkeypatch.setattr(promote_model_channel, "verify_manifest", lambda _bucket, key: key.rsplit("/", 2)[-2])

    result = promote_model_channel.promote(client, "models", "models/pp-ocrv6-small/channels/candidate.json", "models/pp-ocrv6-small/channels/stable.json")

    assert result["promoted"] is True
    assert len(client.puts) == 1
    stable = json.loads(client.puts[0]["Body"].decode())
    assert stable["manifest_key"].endswith("candidate-v2/manifest.json")
    assert stable["previous_manifest_key"].endswith("stable-v1/manifest.json")


def test_rollback_requires_history_and_verifies_selected_manifest(monkeypatch) -> None:
    previous = "models/pp-ocrv6-small/stable-v0/manifest.json"
    client = _objects([{"manifest_key": previous, "action": "promote"}])
    client.objects[previous] = _manifest("stable-v0")
    monkeypatch.setattr(rollback_model_channel, "verify_manifest", lambda _bucket, key: key.rsplit("/", 2)[-2])

    result = rollback_model_channel.rollback(client, "models", "models/pp-ocrv6-small/channels/stable.json", previous)

    assert result["rolled_back"] is True
    assert json.loads(client.puts[0]["Body"].decode())["manifest_key"] == previous

    with pytest.raises(ValueError, match="previously verified"):
        rollback_model_channel.rollback(client, "models", "models/pp-ocrv6-small/channels/stable.json", "models/pp-ocrv6-small/unknown/manifest.json")


def _compare(evidence: dict[str, object], stable_key: str | None = "models/pp-ocrv6-small/stable-v1/manifest.json") -> dict[str, object]:
    candidate = json.loads(_manifest("candidate-v2", evidence))
    stable = json.loads(_manifest("stable-v1"))
    return compare_manifests("models/pp-ocrv6-small/candidate-v2/manifest.json", candidate, stable_key, stable)


def _comparison(evidence: dict[str, object]) -> dict[str, object]:
    return evidence["evaluation"]["comparison"]  # type: ignore[index, return-value]


def test_paired_comparison_decides_and_reports_the_figures_measured_together() -> None:
    report = _compare(_evidence())

    assert report["eligible"] is True
    assert report["comparison"]["field_accuracy"] == {"candidate": 0.98, "stable": 0.97, "delta": pytest.approx(0.01)}
    assert report["comparison"]["holdout_accuracy"]["delta"] == pytest.approx(0.02)
    assert report["comparison"]["critical_field_deltas"]["map_name"]["delta"] == pytest.approx(0.025)


def test_a_failed_paired_comparison_blocks_promotion_with_its_own_reasons() -> None:
    evidence = _evidence()
    _comparison(evidence).update(verdict="failed", reasons=["regresses duration_seconds: 35/40 vs 36/40 on fixtures"])

    report = _compare(evidence)

    assert report["eligible"] is False
    assert "regresses duration_seconds: 35/40 vs 36/40 on fixtures" in report["reasons"]


def test_missing_paired_comparison_blocks_promotion() -> None:
    evidence = _evidence()
    del evidence["evaluation"]["comparison"]  # type: ignore[attr-defined, index]

    assert "missing paired comparison evidence; evaluate the candidate again" in _compare(evidence)["reasons"]


def test_a_comparison_against_a_model_no_longer_in_service_is_stale() -> None:
    report = _compare(_evidence(), stable_key="models/pp-ocrv6-small/stable-v2/manifest.json")

    assert report["eligible"] is False
    assert any("different model than the one in service" in reason for reason in report["reasons"])


def test_a_candidate_evaluated_without_a_baseline_may_become_the_first_stable() -> None:
    evidence = _evidence()
    _comparison(evidence).update(verdict="no_baseline", stable=None)

    assert _compare(evidence, stable_key=None)["eligible"] is True
    assert _compare(evidence)["eligible"] is False  # a stable exists now, so the comparison is stale


def test_run_code_fixtures_must_be_exact_regardless_of_the_comparison() -> None:
    evidence = _evidence()
    evidence["evaluation"]["fixture"]["run_code"] = {"field_accuracy": 0.9}  # type: ignore[index]

    assert "run-code fixture accuracy is below the release gate" in _compare(evidence)["reasons"]
