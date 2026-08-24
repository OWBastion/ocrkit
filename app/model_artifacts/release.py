from __future__ import annotations

import json
from typing import Any

from .constants import MODEL_OBJECT_PREFIX

STABLE_CHANNEL_KEY = f"{MODEL_OBJECT_PREFIX}/channels/stable.json"
CANDIDATE_CHANNEL_KEY = f"{MODEL_OBJECT_PREFIX}/channels/candidate.json"
MIN_FIXTURE_FIELD_ACCURACY = 0.9604221635883905
MIN_RUN_CODE_ACCURACY = 1.0


def validate_channel_key(channel_key: Any, *, allow_stable: bool = False) -> str:
    if not isinstance(channel_key, str):
        raise ValueError("model release channel key is invalid")
    expected_prefix = f"{MODEL_OBJECT_PREFIX}/channels/"
    if not channel_key.startswith(expected_prefix) or not channel_key.endswith(".json"):
        raise ValueError("model release channel key is invalid")
    if not allow_stable and channel_key == STABLE_CHANNEL_KEY:
        raise ValueError("stable channel requires explicit promotion")
    return channel_key


def parse_channel(payload: bytes, channel_key: str) -> dict[str, Any]:
    validate_channel_key(channel_key, allow_stable=True)
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model release channel must be valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("model") != "pp-ocrv6-small":
        raise ValueError("model release channel has an unsupported schema")
    manifest_key = data.get("manifest_key")
    if not isinstance(manifest_key, str) or not manifest_key.startswith(f"{MODEL_OBJECT_PREFIX}/") or not manifest_key.endswith("/manifest.json"):
        raise ValueError("model release channel manifest key is invalid")
    return data


def parse_manifest(payload: bytes, manifest_key: str) -> dict[str, Any]:
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model manifest must be valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("model") != "pp-ocrv6-small":
        raise ValueError("model manifest has an unsupported schema")
    version = data.get("version")
    if not isinstance(version, str) or not version or "/" in version:
        raise ValueError("model manifest version is invalid")
    if manifest_key != f"{MODEL_OBJECT_PREFIX}/{version}/manifest.json":
        raise ValueError("model manifest key does not match its version")
    return data


def _status(value: Any) -> str:
    return value.get("status", "missing") if isinstance(value, dict) else "missing"


def _metric(report: Any, name: str) -> float | None:
    if not isinstance(report, dict):
        return None
    value = report.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _field_metrics(report: Any) -> dict[str, float]:
    if not isinstance(report, dict) or not isinstance(report.get("field_metrics"), dict):
        return {}
    result: dict[str, float] = {}
    for name, item in report["field_metrics"].items():
        accuracy = item.get("accuracy") if isinstance(item, dict) else None
        if isinstance(name, str) and isinstance(accuracy, (int, float)):
            result[name] = float(accuracy)
    return result


def compare_manifests(
    candidate_manifest_key: str,
    candidate_manifest: dict[str, Any],
    stable_manifest_key: str | None,
    stable_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_evidence = candidate_manifest.get("release_evidence")
    candidate_evidence = candidate_evidence if isinstance(candidate_evidence, dict) else {}
    evaluation = candidate_evidence.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    fixture = evaluation.get("fixture")
    holdout = evaluation.get("holdout")
    checks = {
        "fixture": _status(fixture) == "passed",
        "holdout": _status(holdout) == "passed",
        "full_test_suite": _status(candidate_evidence.get("full_test_suite")) == "passed",
        "provenance": _status(candidate_evidence.get("provenance")) == "recorded",
        "compatibility": _status(candidate_evidence.get("compatibility")) == "passed",
    }
    reasons = [f"missing or failing {name} evidence" for name, passed in checks.items() if not passed]
    fixture_accuracy = _metric(fixture, "field_accuracy")
    run_code_accuracy = _metric(fixture.get("run_code") if isinstance(fixture, dict) else None, "field_accuracy")
    if fixture_accuracy is None or fixture_accuracy < MIN_FIXTURE_FIELD_ACCURACY:
        reasons.append("fixture field accuracy is below the release gate")
    if run_code_accuracy is None or run_code_accuracy < MIN_RUN_CODE_ACCURACY:
        reasons.append("run-code fixture accuracy is below the release gate")

    stable_evidence = stable_manifest.get("release_evidence", {}) if isinstance(stable_manifest, dict) else {}
    stable_eval = stable_evidence.get("evaluation", {}) if isinstance(stable_evidence, dict) else {}
    stable_fixture = stable_eval.get("fixture") if isinstance(stable_eval, dict) else None
    candidate_fields = _field_metrics(fixture)
    stable_fields = _field_metrics(stable_fixture)
    field_deltas = {
        name: {
            "candidate": candidate_fields[name],
            "stable": stable_fields[name],
            "delta": candidate_fields[name] - stable_fields[name],
        }
        for name in sorted(candidate_fields.keys() & stable_fields.keys())
    }

    candidate_accuracy = _metric(fixture, "field_accuracy")
    stable_accuracy = _metric(stable_fixture, "field_accuracy")
    candidate_run_code = _metric(fixture.get("run_code") if isinstance(fixture, dict) else None, "field_accuracy")
    stable_run_code = _metric(stable_fixture.get("run_code") if isinstance(stable_fixture, dict) else None, "field_accuracy")
    candidate_errors = _metric(fixture, "false_confident_errors")
    stable_errors = _metric(stable_fixture, "false_confident_errors")
    if candidate_errors is not None and stable_errors is not None and candidate_errors > stable_errors:
        reasons.append("candidate has more false-confident errors than stable")
    for name, delta in field_deltas.items():
        if delta["delta"] < 0:
            reasons.append(f"candidate regresses critical field {name}")
    if candidate_accuracy is not None and stable_accuracy is not None and candidate_accuracy < stable_accuracy:
        reasons.append("candidate fixture accuracy is below stable")
    if candidate_run_code is not None and stable_run_code is not None and candidate_run_code < stable_run_code:
        reasons.append("candidate run-code accuracy is below stable")

    return {
        "schema_version": 1,
        "eligible": not reasons,
        "reasons": reasons,
        "candidate": {
            "manifest_key": candidate_manifest_key,
            "version": candidate_manifest["version"],
            "evidence": candidate_evidence,
        },
        "stable": {
            "manifest_key": stable_manifest_key,
            "version": stable_manifest.get("version") if isinstance(stable_manifest, dict) else None,
            "evidence": stable_evidence,
        },
        "comparison": {
            "field_accuracy": {"candidate": candidate_accuracy, "stable": stable_accuracy, "delta": candidate_accuracy - stable_accuracy if candidate_accuracy is not None and stable_accuracy is not None else None},
            "run_code_accuracy": {"candidate": candidate_run_code, "stable": stable_run_code, "delta": candidate_run_code - stable_run_code if candidate_run_code is not None and stable_run_code is not None else None},
            "critical_field_deltas": field_deltas,
            "false_confident_errors": {"candidate": candidate_errors, "stable": stable_errors},
        },
    }
