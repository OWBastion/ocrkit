"""Experiment runner and metrics for the #4 adjudication comparison.

Compares at least:

1. deterministic normalization only (arm A);
2. deterministic normalization plus the candidate adjudicator (arm B), evaluated
   only on the residual unresolved records.

All outcomes are derived from stored input records and captured provider output,
so results are replayable without assuming provider determinism. The go/no-go
decision is a recommendation; maintainers approve the gate.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.parser.terminology import TerminologyCatalog, normalize_roi_text

from .adjudicator import AdjudicationOutput, Adjudicator, HeuristicAdjudicator
from .records import (
    NormalizationResult,
    ReviewedAnnotationRecord,
    canonicalize,
    compute_input_digest,
    primary_engine_text,
)

# Default gate, to be approved by maintainers before any maintained adapter.
DEFAULT_GATE: dict[str, float] = {
    "min_unresolved_reduction": 0.30,
    "max_false_confident_correction_rate": 0.02,
    "min_precision_on_resolved": 0.90,
}

DEFAULT_CONFIDENCE_GATE = 0.9

HEURISTIC_ADJUDICATOR = HeuristicAdjudicator()

_FAILURE_REASON_CODES = {"timeout", "provider_failure", "invalid_output", "candidate_not_in_allowlist"}


@dataclass
class ArmAOutcome:
    record_id: str
    digest: str
    resolved: bool
    decision: str
    candidate: str | None = None


@dataclass
class ArmBOutcome:
    record_id: str
    digest: str
    resolved: bool
    decision: str
    candidate: str | None = None
    confidence: float | None = None
    reason_code: str = ""
    provider: str = ""


@dataclass
class ExperimentResult:
    metrics: dict[str, Any]
    arm_a: list[ArmAOutcome] = field(default_factory=list)
    arm_b: list[ArmBOutcome] = field(default_factory=list)
    captured_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


def _compute_normalization(record: ReviewedAnnotationRecord, catalog: TerminologyCatalog | None) -> NormalizationResult | None:
    if catalog is None:
        return record.normalization
    text = primary_engine_text(record)
    if text is None:
        return record.normalization
    result = normalize_roi_text(text, record.layout_version, record.roi, catalog)
    return NormalizationResult(
        decision=result.decision,
        rules_version=result.rules_version,
        scope_id=result.scope_id,
        normalized_text=result.normalized_text,
    )


def _arm_a_resolution(record: ReviewedAnnotationRecord, normalization: NormalizationResult | None) -> ArmAOutcome:
    digest = compute_input_digest(record)
    if normalization is None:
        return ArmAOutcome(record_id=record.record_id, digest=digest, resolved=False, decision="unresolved")
    normalized = canonicalize(normalization.normalized_text) if normalization.normalized_text else ""
    for candidate in record.candidates:
        if canonicalize(candidate) == normalized:
            return ArmAOutcome(
                record_id=record.record_id,
                digest=digest,
                resolved=True,
                decision=normalization.decision,
                candidate=candidate,
            )
    return ArmAOutcome(record_id=record.record_id, digest=digest, resolved=False, decision=normalization.decision)


def _arm_b_resolution(record: ReviewedAnnotationRecord, output: AdjudicationOutput) -> ArmBOutcome:
    resolved = output.decision == "constrained_match" and output.candidate in record.candidates
    return ArmBOutcome(
        record_id=record.record_id,
        digest=output.input_digest,
        resolved=resolved,
        decision=output.decision,
        candidate=output.candidate,
        confidence=output.confidence,
        reason_code=output.reason_code,
        provider=output.provider,
    )


def _is_correct(record: ReviewedAnnotationRecord, candidate: str | None) -> bool:
    if candidate is None:
        return False
    expected = canonicalize(record.canonical_truth) if record.canonical_truth else canonicalize(record.truth)
    return canonicalize(candidate) == expected


def _cost_per_1000(outputs: list[AdjudicationOutput], provider: Any) -> tuple[float, str]:
    """Estimate cost from captured usage tokens when available."""
    input_tokens = sum(int(out.usage.get("prompt_tokens", 0)) for out in outputs if out.usage and out.usage.get("prompt_tokens"))
    output_tokens = sum(int(out.usage.get("completion_tokens", 0)) for out in outputs if out.usage and out.usage.get("completion_tokens"))
    if input_tokens or output_tokens:
        per_input = getattr(provider, "cost_per_million_input", 0.0)
        per_output = getattr(provider, "cost_per_million_output", 0.0)
        cost = (input_tokens * per_input + output_tokens * per_output) / 1_000_000 * 1000
        return round(cost, 4), "usage_tokens"
    return 0.0, "unmeasured_without_usage"


def run_experiment(
    records: list[ReviewedAnnotationRecord],
    catalog: TerminologyCatalog | None,
    adjudicator: Adjudicator = HEURISTIC_ADJUDICATOR,
    *,
    gate: dict[str, float] | None = None,
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE,
    replay_outputs: dict[str, dict[str, Any]] | None = None,
    capture_output: Callable[[AdjudicationOutput], None] | None = None,
    provider: Any | None = None,
) -> ExperimentResult:
    """Run both arms and return metrics plus per-record outcomes.

    ``replay_outputs`` maps input digest to a previously captured
    ``AdjudicationOutput`` and is used instead of calling the adjudicator, so a
    report can be reproduced without provider calls. ``capture_output`` is
    invoked for every new provider output so results can be stored for replay.
    """
    gate = gate or dict(DEFAULT_GATE)
    arm_a: list[ArmAOutcome] = []
    arm_b: list[ArmBOutcome] = []
    arm_b_records: list[ReviewedAnnotationRecord] = []
    captured: dict[str, dict[str, Any]] = {}
    provider_calls = 0
    replayed_count = 0
    provider_outputs: list[AdjudicationOutput] = []
    failures = 0

    for record in records:
        normalization = _compute_normalization(record, catalog)
        outcome_a = _arm_a_resolution(record, normalization)
        arm_a.append(outcome_a)
        if outcome_a.resolved:
            continue

        digest = compute_input_digest(record)
        replayed = replay_outputs.get(digest) if replay_outputs else None
        if replayed is not None:
            replayed_count += 1
            output = AdjudicationOutput.model_validate(replayed)
        else:
            provider_calls += 1
            output = adjudicator.adjudicate(record)
            captured[digest] = output.model_dump()
            if capture_output is not None:
                capture_output(output)
        if output.usage is not None:
            provider_outputs.append(output)
        if output.reason_code in _FAILURE_REASON_CODES:
            failures += 1
        arm_b.append(_arm_b_resolution(record, output))
        arm_b_records.append(record)

    total = len(records)
    arm_a_resolved = sum(outcome.resolved for outcome in arm_a)
    arm_a_unresolved = total - arm_a_resolved
    adjudicator_inputs = len(arm_b)
    arm_b_resolved = sum(outcome.resolved for outcome in arm_b)
    arm_b_unresolved = adjudicator_inputs - arm_b_resolved

    correct = sum(_is_correct(record, outcome.candidate) for record, outcome in zip(arm_b_records, arm_b) if outcome.resolved)
    wrong = arm_b_resolved - correct
    false_confident = sum(
        _is_correct(record, outcome.candidate) is False
        and outcome.confidence is not None
        and outcome.confidence >= confidence_gate
        for record, outcome in zip(arm_b_records, arm_b)
        if outcome.resolved
    )

    reduction = (arm_a_unresolved - arm_b_unresolved) / arm_a_unresolved if arm_a_unresolved else 0.0
    precision = correct / arm_b_resolved if arm_b_resolved else 0.0
    false_confident_rate = false_confident / arm_b_resolved if arm_b_resolved else 0.0

    coverage: dict[str, dict[str, int]] = {}
    for record, outcome in zip(arm_b_records, arm_b):
        family = record.field_family or record.roi
        entry = coverage.setdefault(family, {"total": 0, "resolved": 0, "correct": 0, "wrong": 0, "unresolved": 0})
        entry["total"] += 1
        if outcome.resolved:
            entry["resolved"] += 1
            if _is_correct(record, outcome.candidate):
                entry["correct"] += 1
            else:
                entry["wrong"] += 1
        else:
            entry["unresolved"] += 1

    cost_per_1000, cost_basis = _cost_per_1000(provider_outputs, provider)
    metrics: dict[str, Any] = {
        "inputs": {
            "records": total,
            "ground_truth_source": "reviewed_annotations",
        },
        "arm_a": {"resolved": arm_a_resolved, "unresolved": arm_a_unresolved},
        "arm_b": {
            "adjudicator_inputs": adjudicator_inputs,
            "resolved": arm_b_resolved,
            "unresolved": arm_b_unresolved,
            "provider_calls": provider_calls,
            "replayed_outputs": replayed_count,
            "failures": failures,
        },
        "metrics": {
            "manual_review_reduction": round(reduction, 4),
            "precision_on_resolved": round(precision, 4),
            "false_corrections": wrong,
            "false_confident_corrections": false_confident,
            "false_confident_rate": round(false_confident_rate, 4),
            "coverage": coverage,
            "cost_per_1000_candidates": cost_per_1000,
            "cost_basis": cost_basis,
            "provider_failure_rate": round(failures / adjudicator_inputs, 4) if adjudicator_inputs else 0.0,
        },
        "gate": gate,
        "decision": _go_no_go(gate, arm_a_unresolved, adjudicator_inputs, reduction, precision, false_confident_rate),
        "adjudicator": {
            "name": adjudicator.name,
            "model": getattr(adjudicator, "model", None),
            "confidence_gate": confidence_gate,
        },
        "notes": [
            "the go/no-go decision is a recommendation for maintainer approval",
            "a real decision requires ground truth from platform-reviewed dataset imports (#5), not fixtures",
        ],
    }
    return ExperimentResult(metrics=metrics, arm_a=arm_a, arm_b=arm_b, captured_outputs=captured)


def _go_no_go(
    gate: dict[str, float],
    arm_a_unresolved: int,
    adjudicator_inputs: int,
    reduction: float,
    precision: float,
    false_confident_rate: float,
) -> str:
    if arm_a_unresolved == 0:
        return "not_applicable"
    if adjudicator_inputs == 0:
        return "insufficient_data"
    if reduction >= gate["min_unresolved_reduction"] and precision >= gate["min_precision_on_resolved"] and false_confident_rate <= gate["max_false_confident_correction_rate"]:
        return "go"
    return "no_go"


def write_report(report_path: Path, result: ExperimentResult, captured_outputs: dict[str, dict[str, Any]]) -> None:
    """Write the JSON report plus captured outputs for replay."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs_dir = report_path.parent / "captured-outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for digest, output in sorted(captured_outputs.items()):
        (outputs_dir / f"{digest}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_replay_outputs(replay_dir: Path) -> dict[str, dict[str, Any]]:
    """Load previously captured adjudicator outputs keyed by input digest."""
    if not replay_dir.is_dir():
        return {}
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(replay_dir.glob("*.json")):
        outputs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return outputs
