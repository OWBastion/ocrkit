from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.parser.terminology import load_terminology_rules
from training.adjudication.adjudicator import (
    AdjudicationOutput,
    HeuristicAdjudicator,
    OpenAICompatibleProvider,
    ProviderAdjudicator,
)
from training.adjudication.evaluate import (
    DEFAULT_GATE,
    load_replay_outputs,
    run_experiment,
    write_report,
)
from training.adjudication.records import (
    EngineCandidate,
    NormalizationResult,
    ReviewedAnnotationRecord,
    canonicalize,
    compute_input_digest,
    load_records,
)

LP_CANDIDATES = ["增益", "减益", "总计"]
CATALOG = load_terminology_rules(Path("configs/terminology.yaml"))


def _record(
    *,
    record_id: str = "r1",
    engines: list[dict[str, object]] | None = None,
    candidates: list[str] | None = None,
    normalization: NormalizationResult | None = None,
    truth: str = "增益",
    canonical_truth: str | None = "增益",
) -> ReviewedAnnotationRecord:
    return ReviewedAnnotationRecord(
        schema_version="1",
        record_id=record_id,
        layout_version="1280x720-v6",
        roi="left_panel",
        field_family="challenge_stats",
        engines=[EngineCandidate(**item) for item in engines or [{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}]],
        candidates=candidates or LP_CANDIDATES,
        candidates_version="1",
        normalization=normalization,
        truth=truth,
        canonical_truth=canonical_truth,
    )


def _unresolved_normalization() -> NormalizationResult:
    return NormalizationResult(decision="unresolved", rules_version="1", scope_id="left_panel.challenge_stats", normalized_text="编益X")


class TestRecords:
    def test_input_digest_is_stable_and_excludes_ground_truth(self) -> None:
        record = _record()
        assert compute_input_digest(record) == compute_input_digest(record)
        changed_truth = _record(truth="减益", canonical_truth="减益")
        assert compute_input_digest(record) == compute_input_digest(changed_truth)

    def test_extra_keys_are_forbidden_for_privacy(self) -> None:
        from pydantic import ValidationError

        payload = _record().model_dump()
        payload["image_url"] = "https://private.example/screenshot.png"
        with pytest.raises(ValidationError):
            ReviewedAnnotationRecord.model_validate(payload)

    def test_digest_mismatch_is_rejected(self) -> None:
        payload = _record().model_dump()
        payload["input_digest"] = "0" * 64
        with pytest.raises(Exception, match="input_digest mismatch"):
            ReviewedAnnotationRecord.model_validate(payload)

    def test_load_records_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "records.jsonl"
        record = _record(record_id="round-trip")
        data = record.model_dump()
        data["input_digest"] = compute_input_digest(record)
        path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        loaded = load_records(path)
        assert len(loaded) == 1
        assert loaded[0].record_id == "round-trip"


class TestHeuristicAdjudicator:
    def test_consensus_resolves_to_allowlist_candidate(self) -> None:
        record = _record(
            engines=[
                {"engine": "rapidocr", "text": "编益X", "confidence": 0.92},
                {"engine": "vision", "text": "增益", "confidence": 0.88},
            ],
            normalization=_unresolved_normalization(),
        )
        output = HeuristicAdjudicator().adjudicate(record)
        assert output.decision == "constrained_match"
        assert output.candidate == "增益"
        assert output.reason_code == "heuristic_consensus"
        assert output.provider == "none"

    def test_tie_is_ambiguous(self) -> None:
        record = _record(
            engines=[
                {"engine": "rapidocr", "text": "增益", "confidence": 0.91},
                {"engine": "vision", "text": "减益", "confidence": 0.91},
            ],
            normalization=_unresolved_normalization(),
        )
        output = HeuristicAdjudicator().adjudicate(record)
        assert output.decision == "ambiguous"
        assert output.candidate is None

    def test_no_allowlist_evidence_is_unresolved(self) -> None:
        record = _record(
            engines=[{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}],
            normalization=_unresolved_normalization(),
        )
        output = HeuristicAdjudicator().adjudicate(record)
        assert output.decision == "unresolved"
        assert output.reason_code == "no_evidence"


class TestProviderAdjudicator:
    def _wrapper(self, call_fn, timeout: float = 5.0) -> ProviderAdjudicator:
        return ProviderAdjudicator("fake", call_fn, model="fake-model", prompt_version="v1", timeout_seconds=timeout)

    def test_valid_output_passes_through_and_captures_raw(self) -> None:
        def call_fn(record):
            return AdjudicationOutput(
                decision="constrained_match",
                candidate="增益",
                confidence=0.95,
                reason_code="provider_selected",
                provider="fake",
                input_digest=compute_input_digest(record),
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                raw_output={"response": {"ok": True}},
            )

        record = _record(normalization=_unresolved_normalization())
        output = self._wrapper(call_fn).adjudicate(record)
        assert output.decision == "constrained_match"
        assert output.candidate == "增益"
        assert output.provider == "fake"
        assert output.raw_output == {"response": {"ok": True}}
        assert output.input_digest == compute_input_digest(record)

    def test_exception_is_fail_closed(self) -> None:
        def call_fn(record):
            raise RuntimeError("boom")

        output = self._wrapper(call_fn).adjudicate(_record())
        assert output.decision == "unresolved"
        assert output.reason_code == "provider_failure"

    def test_timeout_is_fail_closed(self) -> None:
        def call_fn(record):
            time.sleep(0.5)
            return AdjudicationOutput(decision="unresolved", reason_code="x", provider="fake", input_digest="")

        output = self._wrapper(call_fn, timeout=0.05).adjudicate(_record())
        assert output.decision == "unresolved"
        assert output.reason_code == "timeout"

    def test_invalid_output_type_is_fail_closed(self) -> None:
        def call_fn(record):
            return {"decision": "constrained_match", "candidate": "增益"}  # not an AdjudicationOutput

        output = self._wrapper(call_fn).adjudicate(_record())
        assert output.reason_code == "invalid_output"

    def test_off_allowlist_candidate_is_rejected(self) -> None:
        def call_fn(record):
            return AdjudicationOutput(
                decision="constrained_match",
                candidate="编造值",
                confidence=0.99,
                reason_code="provider_selected",
                provider="fake",
                input_digest=compute_input_digest(record),
            )

        output = self._wrapper(call_fn).adjudicate(_record(normalization=_unresolved_normalization()))
        assert output.decision == "unresolved"
        assert output.reason_code == "candidate_not_in_allowlist"


class TestOpenAICompatibleProvider:
    def test_prompt_contains_only_record_metadata(self) -> None:
        provider = OpenAICompatibleProvider(endpoint="http://example.test", model="m", api_key="k")
        record = _record(normalization=_unresolved_normalization())
        user_message = json.loads(provider.build_prompt(record)[1]["content"])
        assert user_message["candidates"] == LP_CANDIDATES
        assert user_message["roi"] == "left_panel"
        assert "truth" not in user_message
        assert "canonical_truth" not in user_message
        assert "image" not in json.dumps(user_message)
        assert "url" not in json.dumps(user_message)

    def test_parses_valid_response(self) -> None:
        provider = OpenAICompatibleProvider(endpoint="http://example.test", model="m", api_key="k")
        record = _record(normalization=_unresolved_normalization())
        body = {
            "choices": [{"message": {"content": json.dumps({"decision": "constrained_match", "candidate": "增益", "confidence": 0.95, "reason_code": "provider_selected"})}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
        output = provider._adjudicate_from_response(record, body)
        assert output.decision == "constrained_match"
        assert output.candidate == "增益"
        assert output.usage == {"prompt_tokens": 12, "completion_tokens": 6}

    def test_missing_content_raises(self) -> None:
        provider = OpenAICompatibleProvider(endpoint="http://example.test", model="m", api_key="k")
        with pytest.raises(ValueError):
            provider._adjudicate_from_response(_record(), {"choices": []})


class TestExperiment:
    def test_replay_does_not_call_provider(self) -> None:
        record = _record(engines=[{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}], normalization=_unresolved_normalization())
        calls: list[str] = []

        class CountingAdjudicator:
            name = "counting"

            def adjudicate(self, record):
                calls.append(record.record_id)
                return AdjudicationOutput(decision="unresolved", reason_code="no_evidence", provider="counting", input_digest=compute_input_digest(record))

        first = run_experiment([record], CATALOG, CountingAdjudicator())
        assert len(calls) == 1
        replay = {digest: output for digest, output in first.captured_outputs.items()}
        second = run_experiment([record], CATALOG, CountingAdjudicator(), replay_outputs=replay)
        assert len(calls) == 1  # no additional provider call
        assert second.metrics["arm_b"]["replayed_outputs"] == 1
        assert second.metrics["arm_b"]["provider_calls"] == 0

    def test_metrics_measure_residual_reduction_and_false_corrections(self) -> None:
        records = [
            # resolved by deterministic normalization -> not sent to the adjudicator
            _record(record_id="resolved-by-normalization", engines=[{"engine": "rapidocr", "text": "编益", "confidence": 0.93}]),
            # residual, adjudicator resolves correctly
            _record(
                record_id="resolved-by-adjudicator",
                engines=[{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}, {"engine": "vision", "text": "增益", "confidence": 0.88}],
                normalization=_unresolved_normalization(),
            ),
            # residual, adjudicator resolves wrongly (false confident correction)
            _record(
                record_id="wrong-resolution",
                engines=[{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}, {"engine": "vision", "text": "编益Y", "confidence": 0.99}],
                normalization=_unresolved_normalization(),
                truth="增益",
                canonical_truth="增益",
            ),
            # residual, adjudicator abstains
            _record(
                record_id="still-unresolved",
                engines=[{"engine": "rapidocr", "text": "编益X", "confidence": 0.92}],
                normalization=_unresolved_normalization(),
            ),
        ]

        class FakeAdjudicator:
            name = "fake"

            def adjudicate(self, record):
                if record.record_id == "resolved-by-adjudicator":
                    candidate, decision = "增益", "constrained_match"
                elif record.record_id == "wrong-resolution":
                    candidate, decision = "减益", "constrained_match"
                else:
                    candidate, decision = None, "unresolved"
                return AdjudicationOutput(
                    decision=decision,
                    candidate=candidate,
                    confidence=0.99 if candidate else None,
                    reason_code="fake",
                    provider="fake",
                    input_digest=compute_input_digest(record),
                )

        result = run_experiment(records, CATALOG, FakeAdjudicator())
        metrics = result.metrics
        assert metrics["arm_a"] == {"resolved": 1, "unresolved": 3}
        assert metrics["arm_b"]["adjudicator_inputs"] == 3
        assert metrics["arm_b"]["resolved"] == 2
        assert metrics["arm_b"]["unresolved"] == 1
        assert metrics["metrics"]["manual_review_reduction"] == pytest.approx(2 / 3, abs=5e-5)
        assert metrics["metrics"]["precision_on_resolved"] == pytest.approx(0.5)
        assert metrics["metrics"]["false_corrections"] == 1
        assert metrics["metrics"]["false_confident_corrections"] == 1
        assert metrics["metrics"]["false_confident_rate"] == pytest.approx(0.5)
        assert metrics["metrics"]["coverage"]["challenge_stats"]["correct"] == 1
        assert metrics["metrics"]["coverage"]["challenge_stats"]["wrong"] == 1
        assert metrics["decision"] == "no_go"

    def test_report_round_trip(self, tmp_path: Path) -> None:
        record = _record(normalization=_unresolved_normalization())
        result = run_experiment([record], CATALOG, HeuristicAdjudicator())
        report_path = tmp_path / "report.json"
        write_report(report_path, result, result.captured_outputs)
        assert report_path.is_file()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["inputs"]["records"] == 1
        captured = load_replay_outputs(tmp_path / "captured-outputs")
        assert len(captured) == 1
        assert next(iter(captured.values()))["decision"] in {"constrained_match", "ambiguous", "unresolved"}


def test_fixture_is_valid_and_experiment_runs() -> None:
    fixture = Path("training/adjudication/fixtures/reviewed_annotations.jsonl")
    assert fixture.is_file()
    records = load_records(fixture)
    assert len(records) >= 9
    result = run_experiment(records, CATALOG, HeuristicAdjudicator())
    metrics = result.metrics
    assert metrics["inputs"]["records"] == len(records)
    assert metrics["arm_a"]["resolved"] + metrics["arm_a"]["unresolved"] == len(records)
    assert metrics["arm_b"]["adjudicator_inputs"] == metrics["arm_a"]["unresolved"]
    assert metrics["metrics"]["precision_on_resolved"] in {0.0, 1.0}
    assert metrics["decision"] in {"go", "no_go", "not_applicable", "insufficient_data"}
    assert canonicalize("增益") == "增益"
    assert DEFAULT_GATE["min_unresolved_reduction"] > 0
