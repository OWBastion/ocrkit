"""Evaluation for the #25 Jev-Omni pre-review sidecar experiment.

Runs the bounded decision task over the residual rows the deterministic
auto-accept/auto-reject rules did not decide, fits a routing threshold on a
deterministic source-level fit split, and reports review reduction,
auto-decision accuracy, false-confident errors, manual-transcription share,
and latency/memory per ROI family and per image-token budget on the held-out
validation split. Captured outputs are replayable without the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .records import (
    OPTION_NONE_CORRECT,
    OPTION_NOT_VALID,
    PreReviewRecord,
    canonicalize,
)
from .runner import DecisionOutput, Runner

DEFAULT_GATE: dict[str, float] = {
    "min_review_reduction": 0.25,
    "max_false_confident_rate": 0.02,
    "min_auto_decision_accuracy": 0.97,
}

THRESHOLD_GRID = [round(0.2 + 0.005 * step, 3) for step in range(160)]

TruthKind = Literal["accept_listed", "accept_unlisted", "reject", "unreviewed"]
Route = Literal["auto_accept", "auto_reject", "review"]


def truth_kind(record: PreReviewRecord) -> tuple[TruthKind, int | None]:
    """Map the human review outcome to the option that should be selected."""
    if record.truth_status == "rejected":
        return "reject", len(record.options) - 1
    if record.truth_status != "accepted" or not record.truth_transcription:
        return "unreviewed", None
    target = canonicalize(record.truth_transcription)
    for index, option in enumerate(record.candidate_options):
        if canonicalize(option) == target:
            return "accept_listed", index
    return "accept_unlisted", len(record.options) - 2


def effective_confidence(output: DecisionOutput, temperature: float | None = None) -> float | None:
    """Confidence of the selected option after optional temperature scaling."""
    if output.selected_index is None or not output.probabilities:
        return None
    probs = [float(output.probabilities.get(option, 0.0)) for option in output.options]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [value / total for value in probs]
    if temperature:
        probs = _temperature_scale(probs, temperature)
    return probs[output.selected_index]


def route(output: DecisionOutput, threshold: float, temperature: float | None = None) -> tuple[Route, int | None]:
    """Map a decision output to the pre-review routing action."""
    if output.status != "ok" or output.selected_index is None:
        return "review", None
    confidence = effective_confidence(output, temperature)
    if confidence is None:
        return "review", None
    option = output.options[output.selected_index]
    if option == OPTION_NONE_CORRECT or confidence < threshold:
        return "review", None
    if option == OPTION_NOT_VALID:
        return "auto_reject", output.selected_index
    return "auto_accept", output.selected_index


def _is_correct(record: PreReviewRecord, option_index: int | None) -> bool:
    if option_index is None:
        return False
    _, correct = truth_kind(record)
    return correct is not None and option_index == correct


def split_assign(record: PreReviewRecord, fit_fraction: float) -> Literal["fit", "validation"]:
    """Deterministic source-level split so crops of one screenshot stay together."""
    value = int(hashlib.sha256(record.source_id.encode("utf-8")).hexdigest()[:16], 16) / 2**64
    return "fit" if value < fit_fraction else "validation"


def _temperature_scale(probabilities: list[float], temperature: float) -> list[float]:
    logits = [math.log(max(value, 1e-12)) / temperature for value in probabilities]
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def _fit_temperature(items: list[tuple[PreReviewRecord, DecisionOutput]]) -> float | None:
    """Fit one global temperature on the correct-option NLL of the fit split."""
    scored: list[tuple[list[float], int]] = []
    for record, output in items:
        _, correct = truth_kind(record)
        if correct is None or not output.probabilities:
            continue
        probs = [float(output.probabilities.get(option, 0.0)) for option in output.options]
        if len(probs) != len(output.options) or sum(probs) <= 0:
            continue
        total = sum(probs)
        scored.append(([value / total for value in probs], correct))
    if len(scored) < 20:
        return None
    best_temperature, best_nll = None, math.inf
    for step in range(1, 101):
        temperature = 0.25 + step * 0.0375
        nll = -sum(math.log(max(_temperature_scale(probs, temperature)[correct], 1e-12)) for probs, correct in scored)
        if nll < best_nll:
            best_temperature, best_nll = temperature, nll
    return best_temperature


def _ece(items: list[tuple[PreReviewRecord, DecisionOutput]], temperature: float | None, bins: int = 10) -> float | None:
    scored: list[tuple[float, bool]] = []
    for record, output in items:
        _, correct = truth_kind(record)
        if correct is None or output.status != "ok" or not output.probabilities:
            continue
        probs = [float(output.probabilities.get(option, 0.0)) for option in output.options]
        if len(probs) != len(output.options) or sum(probs) <= 0:
            continue
        probs = [value / sum(probs) for value in probs]
        if temperature:
            probs = _temperature_scale(probs, temperature)
        top = max(range(len(probs)), key=probs.__getitem__)
        scored.append((probs[top], top == correct))
    if not scored:
        return None
    error = 0.0
    for bucket in range(bins):
        members = [(conf, ok) for conf, ok in scored if bucket / bins <= conf < (bucket + 1) / bins]
        if not members:
            continue
        mean_conf = sum(conf for conf, _ in members) / len(members)
        accuracy = sum(ok for _, ok in members) / len(members)
        error += (len(members) / len(scored)) * abs(mean_conf - accuracy)
    return round(error, 4)


def _metrics_at(
    items: list[tuple[PreReviewRecord, DecisionOutput]],
    threshold: float,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Routing metrics over scored rows at one confidence threshold."""
    total = 0
    auto_accept = auto_reject = review = 0
    correct = false_accept = false_reject = 0
    unlisted = unlisted_flagged = 0
    per_roi: dict[str, dict[str, int]] = {}
    for record, output in items:
        kind, _ = truth_kind(record)
        if kind == "unreviewed":
            continue
        total += 1
        action, option_index = route(output, threshold, temperature)
        entry = per_roi.setdefault(
            record.roi,
            {"total": 0, "auto_accept": 0, "auto_reject": 0, "review": 0, "correct": 0, "false_accept": 0, "false_reject": 0, "unlisted": 0},
        )
        entry["total"] += 1
        if kind == "accept_unlisted":
            unlisted += 1
            entry["unlisted"] += 1
            if output.status == "ok" and output.selected_option == OPTION_NONE_CORRECT:
                unlisted_flagged += 1
        if action == "auto_accept":
            auto_accept += 1
            entry["auto_accept"] += 1
        elif action == "auto_reject":
            auto_reject += 1
            entry["auto_reject"] += 1
        else:
            review += 1
            entry["review"] += 1
        if action == "review":
            continue
        if _is_correct(record, option_index):
            correct += 1
            entry["correct"] += 1
        elif action == "auto_accept":
            false_accept += 1
            entry["false_accept"] += 1
        else:
            false_reject += 1
            entry["false_reject"] += 1
    auto_total = auto_accept + auto_reject
    return {
        "rows": total,
        "auto_accept": auto_accept,
        "auto_reject": auto_reject,
        "review": review,
        "review_reduction": round(auto_total / total, 4) if total else 0.0,
        "auto_decision_accuracy": round(correct / auto_total, 4) if auto_total else None,
        "false_confident_accept": false_accept,
        "false_confident_reject": false_reject,
        "false_confident_rate": round((false_accept + false_reject) / auto_total, 4) if auto_total else None,
        "manual_transcription_rows": unlisted,
        "manual_transcription_share": round(unlisted / total, 4) if total else 0.0,
        "manual_transcription_flagged": unlisted_flagged,
        "per_roi": per_roi,
    }


def _latency_metrics(outputs: list[DecisionOutput]) -> dict[str, Any]:
    latencies = [output.latency_ms for output in outputs if output.latency_ms is not None]
    memories = [
        output.metrics["peak_metal_memory_gb"]
        for output in outputs
        if output.metrics and isinstance(output.metrics.get("peak_metal_memory_gb"), (int, float))
    ]
    input_tokens = [
        output.metrics["input_tokens"]
        for output in outputs
        if output.metrics and isinstance(output.metrics.get("input_tokens"), (int, float))
    ]
    return {
        "decisions": len(outputs),
        "errors": sum(output.status != "ok" for output in outputs),
        "latency_median_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
        "peak_metal_memory_gb": round(max(memories), 2) if memories else None,
        "input_tokens_median": statistics.median(input_tokens) if input_tokens else None,
    }


def fit_threshold(
    items: list[tuple[PreReviewRecord, DecisionOutput]],
    gate: dict[str, float],
    temperature: float | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Lowest threshold on the fit split that satisfies the false-confidence gate."""
    curve: list[dict[str, Any]] = []
    chosen: float | None = None
    for threshold in THRESHOLD_GRID:
        metrics = _metrics_at(items, threshold, temperature)
        auto_total = metrics["auto_accept"] + metrics["auto_reject"]
        feasible = (
            auto_total > 0
            and metrics["false_confident_rate"] is not None
            and metrics["auto_decision_accuracy"] is not None
            and metrics["false_confident_rate"] <= gate["max_false_confident_rate"]
            and metrics["auto_decision_accuracy"] >= gate["min_auto_decision_accuracy"]
        )
        curve.append(
            {
                "threshold": threshold,
                "auto_decided": auto_total,
                "review_reduction": metrics["review_reduction"],
                "false_confident_rate": metrics["false_confident_rate"],
                "auto_decision_accuracy": metrics["auto_decision_accuracy"],
                "feasible": feasible,
            }
        )
        if feasible and chosen is None:
            chosen = threshold
    return chosen, curve


@dataclass
class ExperimentResult:
    """Report payload plus the captured outputs used to produce it."""

    metrics: dict[str, Any]
    outputs: dict[tuple[str, int], DecisionOutput] = field(default_factory=dict)


def run_experiment(
    records: list[PreReviewRecord],
    crop_paths: dict[str, Path],
    runner: Runner,
    *,
    image_tokens_list: list[int],
    gate: dict[str, float] | None = None,
    fit_fraction: float = 0.5,
    replay_outputs: dict[tuple[str, int], dict[str, Any]] | None = None,
    max_rows: int | None = None,
    progress=None,
) -> ExperimentResult:
    """Run residual rows through the runner and produce the report payload."""
    gate = gate or dict(DEFAULT_GATE)
    residual = [record for record in records if record.residual]
    if max_rows is not None and len(residual) > max_rows:
        step = math.ceil(len(residual) / max_rows)
        residual = residual[::step]

    outputs: dict[tuple[str, int], DecisionOutput] = {}
    calls = 0
    replayed = 0
    try:
        for record in residual:
            image_path = crop_paths.get(record.record_id) or Path(record.crop)
            for tokens in image_tokens_list:
                key = (record.input_digest or "", tokens)
                replayed_output = replay_outputs.get(key) if replay_outputs else None
                if replayed_output is not None:
                    replayed += 1
                    outputs[key] = DecisionOutput.model_validate(replayed_output)
                    continue
                output = runner.decide(record, image_path, tokens)
                outputs[key] = output
                calls += 1
                if progress:
                    progress(record, output)
    finally:
        runner.close()

    budgets: dict[str, Any] = {}
    for tokens in image_tokens_list:
        scored = [(record, outputs[(record.input_digest or "", tokens)]) for record in residual if (record.input_digest or "", tokens) in outputs]
        fit_items = [item for item in scored if split_assign(item[0], fit_fraction) == "fit"]
        validation_items = [item for item in scored if split_assign(item[0], fit_fraction) == "validation"]
        temperature = _fit_temperature(fit_items)
        threshold, curve = fit_threshold(fit_items, gate, temperature)
        raw_threshold, raw_curve = fit_threshold(fit_items, gate)
        budget_outputs = [output for _, output in scored]
        pending_items = [item for item in scored if truth_kind(item[0])[0] == "unreviewed"]
        pending_routing: dict[str, int] = {"auto_accept": 0, "auto_reject": 0, "review": 0}
        if threshold is not None:
            for record, output in pending_items:
                action, _ = route(output, threshold, temperature)
                pending_routing[action] += 1
        budgets[str(tokens)] = {
            "fit_rows": len(fit_items),
            "validation_rows": len(validation_items),
            "unreviewed_rows": len(pending_items),
            "threshold": threshold,
            "threshold_curve": curve,
            "temperature_fit": temperature,
            "ece_raw": _ece(validation_items, None),
            "ece_temperature_scaled": _ece(validation_items, temperature),
            "validation": _metrics_at(validation_items, threshold, temperature) if threshold is not None else _metrics_at(validation_items, math.inf, temperature),
            "fit_at_threshold": _metrics_at(fit_items, threshold, temperature) if threshold is not None else None,
            "unreviewed_routing": pending_routing if threshold is not None else None,
            "raw_confidence_reference": {
                "threshold": raw_threshold,
                "threshold_curve": raw_curve,
                "validation": _metrics_at(validation_items, raw_threshold) if raw_threshold is not None else _metrics_at(validation_items, math.inf),
            },
            "resources": _latency_metrics(budget_outputs),
        }

    feasible_budgets = {
        tokens: data
        for tokens, data in budgets.items()
        if data["threshold"] is not None
        and data["validation"]["review_reduction"] >= gate["min_review_reduction"]
    }
    if feasible_budgets:
        best_tokens, _ = max(feasible_budgets.items(), key=lambda item: item[1]["validation"]["review_reduction"])
        recommendation = "keep_sidecar"
        recommended_budget = int(best_tokens)
    elif all(data["threshold"] is None for data in budgets.values()):
        recommendation = "remove"
        recommended_budget = None
    else:
        recommendation = "remove"
        recommended_budget = None
    if not any(data["fit_rows"] for data in budgets.values()):
        recommendation = "insufficient_data"

    report = {
        "schema_version": "1",
        "issue": "ocrkit#25",
        "inputs": {
            "records": len(records),
            "residual_rows": len(residual),
            "scored_rows": sum(record.truth_status is not None for record in residual),
            "unreviewed_rows": sum(record.truth_status is None for record in residual),
            "batches": sorted({record.batch_id for record in residual}),
            "split_rule": f"sha256(source_id) fit_fraction={fit_fraction}",
        },
        "runner": {
            "name": runner.name,
            # Decision outputs carry their own model identity so replayed
            # reports keep provenance of the original run.
            "models": sorted({output.model for output in outputs.values() if output.model}),
            "model_revisions": sorted({output.model_revision for output in outputs.values() if output.model_revision}),
            "prompt_versions": sorted({record.prompt_version for record in residual}),
            "image_tokens": image_tokens_list,
            "model_calls": calls,
            "replayed_outputs": replayed,
        },
        "gate": gate,
        "budgets": budgets,
        "recommendation": {
            "outcome": recommendation,
            "image_tokens": recommended_budget,
            "note": "experiment recommendation; the lifecycle decision is a maintainer call per the issue decision gate",
        },
    }
    return ExperimentResult(metrics=report, outputs=outputs)


def write_report(report_dir: Path, result: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_decisions(path: Path, outputs: dict[tuple[str, int], DecisionOutput] | dict[tuple[str, int], dict[str, Any]]) -> None:
    """Persist captured decisions for replay; keyed by (input_digest, image_tokens)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (digest, tokens), output in sorted(outputs.items()):
            payload = output.model_dump() if isinstance(output, DecisionOutput) else output
            handle.write(json.dumps({"input_digest": digest, "image_tokens": tokens, "output": payload}, ensure_ascii=False) + "\n")


def load_decisions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    outputs: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return outputs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            outputs[(row["input_digest"], int(row["image_tokens"]))] = row["output"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # tolerate a truncated final line from an interrupted run
    return outputs
