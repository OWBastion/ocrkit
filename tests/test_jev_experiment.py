from __future__ import annotations



from training.jev.evaluate import (
    route,
    split_assign,
)
from training.jev.records import (
    OPTION_NONE_CORRECT,
    OPTION_NOT_VALID,
    PreReviewRecord,
    compute_input_digest,
)
from training.jev.runner import DecisionOutput

def _record(**overrides) -> PreReviewRecord:
    payload = {
        "schema_version": "1",
        "record_id": "b1:train:x.png",
        "batch_id": "b1",
        "source_id": "source-a",
        "split": "train",
        "roi": "left_panel",
        "layout_version": "1280x720-v6",
        "crop": "images/train/source-a/left_panel/000.png",
        "crop_sha256": "0" * 64,
        "engines": [{"engine": "rapidocr", "text": "增益", "confidence": 0.9}],
        "options": ["增益", "减益", OPTION_NONE_CORRECT, OPTION_NOT_VALID],
        "truth_status": "accepted",
        "truth_transcription": "增益",
    }
    payload.update(overrides)
    return PreReviewRecord(**payload)


def _output(record: PreReviewRecord, selected: int, confidence: float = 0.99, status: str = "ok") -> DecisionOutput:
    probabilities = [(1.0 - confidence) / (len(record.options) - 1)] * len(record.options)
    probabilities[selected] = confidence
    return DecisionOutput(
        record_id=record.record_id,
        input_digest=compute_input_digest(record),
        runner="test",
        status=status,
        reason_code="ok" if status == "ok" else "worker_failure",
        image_tokens=70,
        options=list(record.options),
        selected_index=selected if status == "ok" else None,
        selected_option=record.options[selected] if status == "ok" else None,
        confidence=confidence if status == "ok" else None,
        probabilities=dict(zip(record.options, probabilities)) if status == "ok" else None,
    )


def test_route_threshold_and_fixed_options() -> None:
    record = _record()
    assert route(_output(record, 0), 0.9) == ("auto_accept", 0)
    assert route(_output(record, 0, confidence=0.5), 0.9) == ("review", None)
    assert route(_output(record, len(record.options) - 1), 0.9) == ("auto_reject", len(record.options) - 1)
    # The none-correct option always routes to human review, whatever its confidence.
    assert route(_output(record, len(record.options) - 2), 0.9) == ("review", None)
    assert route(_output(record, 0, status="error"), 0.9) == ("review", None)


def test_split_assign_is_source_grouped() -> None:
    first = _record(source_id="same-source")
    second = _record(source_id="same-source", record_id="b1:train:y.png")
    assert split_assign(first, 0.5) == split_assign(second, 0.5)
