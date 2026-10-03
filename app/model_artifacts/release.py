from __future__ import annotations

import json
from typing import Any

from .constants import MODEL_OBJECT_PREFIX

STABLE_CHANNEL_KEY = f"{MODEL_OBJECT_PREFIX}/channels/stable.json"
CANDIDATE_CHANNEL_KEY = f"{MODEL_OBJECT_PREFIX}/channels/candidate.json"
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


def _accuracy(item: Any) -> float | None:
    if isinstance(item, dict) and isinstance(item.get("total"), (int, float)) and item["total"]:
        return float(item["matched"]) / float(item["total"])
    return None


def _paired(comparison: Any) -> dict[str, Any]:
    """Candidate and in-service figures that were measured together, under the same code and inputs."""
    comparison = comparison if isinstance(comparison, dict) else {}
    candidate = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    baseline = comparison.get("baseline") if isinstance(comparison.get("baseline"), dict) else {}

    def pick(side: dict[str, Any], *path: str) -> float | None:
        value: Any = side
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return float(value) if isinstance(value, (int, float)) else None

    def pair(*path: str) -> dict[str, float | None]:
        now, before = pick(candidate, *path), pick(baseline, *path)
        return {"candidate": now, "stable": before, "delta": now - before if now is not None and before is not None else None}

    deltas: dict[str, Any] = {}
    now_fields = (candidate.get("fixture") or {}).get("field_metrics", {}) if isinstance(candidate.get("fixture"), dict) else {}
    before_fields = (baseline.get("fixture") or {}).get("field_metrics", {}) if isinstance(baseline.get("fixture"), dict) else {}
    for name in sorted(set(now_fields) & set(before_fields)):
        now, before = _accuracy(now_fields[name]), _accuracy(before_fields[name])
        if now is not None and before is not None:
            deltas[name] = {"candidate": now, "stable": before, "delta": now - before}
    return {
        "field_accuracy": pair("fixture", "field_accuracy"),
        "run_code_accuracy": pair("fixture", "run_code_accuracy"),
        "holdout_accuracy": pair("holdout", "accuracy"),
        "critical_field_deltas": deltas,
    }


def compare_manifests(
    candidate_manifest_key: str,
    candidate_manifest: dict[str, Any],
    stable_manifest_key: str | None,
    stable_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether the candidate may replace the model in service.

    The deciding evidence is the paired comparison recorded when the candidate was evaluated. Numbers stored in the
    two manifests were measured at different times, with different code and fixtures, so they are not compared.
    """
    candidate_evidence = candidate_manifest.get("release_evidence")
    candidate_evidence = candidate_evidence if isinstance(candidate_evidence, dict) else {}
    evaluation = candidate_evidence.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    fixture = evaluation.get("fixture")
    holdout = evaluation.get("holdout")
    comparison = evaluation.get("comparison")
    checks = {
        "fixture": _status(fixture) == "passed",
        "holdout": _status(holdout) == "passed",
        "full_test_suite": _status(candidate_evidence.get("full_test_suite")) == "passed",
        "provenance": _status(candidate_evidence.get("provenance")) == "recorded",
        "compatibility": _status(candidate_evidence.get("compatibility")) == "passed",
    }
    reasons = [f"missing or failing {name} evidence" for name, passed in checks.items() if not passed]
    run_code_accuracy = _metric(fixture.get("run_code") if isinstance(fixture, dict) else None, "field_accuracy")
    if run_code_accuracy is None or run_code_accuracy < MIN_RUN_CODE_ACCURACY:
        reasons.append("run-code fixture accuracy is below the release gate")

    verdict = comparison.get("verdict") if isinstance(comparison, dict) else None
    override = comparison.get("override") if isinstance(comparison, dict) and isinstance(comparison.get("override"), dict) else None
    if verdict not in {"passed", "no_baseline", "failed", "overridden"}:
        reasons.append("missing paired comparison evidence; evaluate the candidate again")
    elif verdict == "failed":
        reasons.extend(str(reason) for reason in comparison.get("reasons") or ["paired comparison failed"])
    elif verdict == "overridden" and not (override and str(override.get("reason", "")).strip()):
        reasons.append("the failed comparison was overridden without a recorded reason")
    if verdict in {"passed", "no_baseline", "failed", "overridden"}:
        compared = (comparison.get("stable") or {}).get("manifest_key") if isinstance(comparison.get("stable"), dict) else None
        if compared != stable_manifest_key:
            reasons.append("the candidate was compared with a different model than the one in service now; evaluate it again")

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
            "evidence": stable_manifest.get("release_evidence", {}) if isinstance(stable_manifest, dict) else {},
        },
        "comparison": _paired(comparison),
        "override": override,
    }
