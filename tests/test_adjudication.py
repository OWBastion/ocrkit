from __future__ import annotations

import json
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
    run_experiment,
)
from training.adjudication.records import (
    EngineCandidate,
    NormalizationResult,
    ReviewedAnnotationRecord,
    compute_input_digest,
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

    def test_extra_keys_are_forbidden_for_privacy(self) -> None:
        from pydantic import ValidationError

        payload = _record().model_dump()
        payload["image_url"] = "https://private.example/screenshot.png"
        with pytest.raises(ValidationError):
            ReviewedAnnotationRecord.model_validate(payload)


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


    def test_exception_is_fail_closed(self) -> None:
        def call_fn(record):
            raise RuntimeError("boom")

        output = self._wrapper(call_fn).adjudicate(_record())
        assert output.decision == "unresolved"
        assert output.reason_code == "provider_failure"


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


class TestExperiment:

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
