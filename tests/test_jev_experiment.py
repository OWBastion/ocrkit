from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.jev.evaluate import (
    DEFAULT_GATE,
    effective_confidence,
    fit_threshold,
    load_decisions,
    route,
    run_experiment,
    split_assign,
    truth_kind,
    write_decisions,
    write_report,
)
from training.jev.records import (
    OPTION_NONE_CORRECT,
    OPTION_NOT_VALID,
    PreReviewRecord,
    compute_input_digest,
    load_records,
    load_studio_records,
    write_records,
)
from training.jev.runner import DecisionOutput, MockRunner, OmniMlxRunner

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "training/jev/omni_worker.py"

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-crop"


def _row(
    crop: str,
    *,
    source: str = "source-a",
    split: str = "train",
    roi: str = "left_panel",
    status: str = "pending",
    rapid: str | None = "增益",
    rapid_conf: float | None = 0.9,
    vision: str | None = "增益",
    vision_conf: float | None = 0.6,
    teacher: str | None = None,
    transcription: str | None = None,
    auto_accept: str | None = None,
    auto_reject: str | None = None,
) -> dict:
    return {
        "crop": crop,
        "source_id": source,
        "split": split,
        "roi": roi,
        "box": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "candidate_text": rapid,
        "confidence": rapid_conf,
        "rapidocr_text": rapid,
        "rapidocr_confidence": rapid_conf,
        "vision_text": vision,
        "vision_confidence": vision_conf,
        "teacher_text": teacher,
        "teacher_confidence": 0.95 if teacher else None,
        "review_status": status,
        "transcription": transcription,
        "auto_accept_reason": auto_accept,
        "auto_reject_reason": auto_reject,
    }


def _batch(tmp_path: Path, rows: list[dict], batch_id: str = "b1") -> Path:
    batch = tmp_path / "batches" / batch_id
    (batch / "dataset" / "review").mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"batch_id": batch_id, "layout_version": "1280x720-v6"}))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)
        crop = batch / "dataset" / row["crop"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(PNG_BYTES + row["crop"].encode())
    for split, split_rows in grouped.items():
        (batch / "dataset" / "review" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
            encoding="utf-8",
        )
    return batch


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


def test_load_studio_records_builds_residual_records(tmp_path: Path) -> None:
    _batch(
        tmp_path,
        [
            _row("images/train/source-a/left_panel/000.png", status="accepted", transcription="增益"),
            _row("images/train/source-a/left_panel/001.png", status="accepted", auto_accept="rapidocr_vision_agreement", transcription="增益"),
            _row("images/train/source-b/left_panel/000.png", source="source-b", status="rejected", auto_reject="run_code.content_mismatch"),
            _row("images/train/source-b/right_panel/000.png", source="source-b", roi="right_panel", status="rejected"),
            _row("images/holdout/source-c/left_panel/000.png", source="source-c", split="holdout", status="pending"),
        ],
    )
    records, crop_paths = load_studio_records(tmp_path / "batches")
    assert len(records) == 5
    assert [r.residual for r in records] == [True, False, False, True, True]
    assert records[1].deterministic == "auto_accept"
    assert records[2].deterministic == "auto_reject"
    assert all(record.input_digest == compute_input_digest(record) for record in records)
    assert set(crop_paths) == {record.record_id for record in records}


def test_options_dedupe_and_fixed_tail(tmp_path: Path) -> None:
    _batch(
        tmp_path,
        [
            _row(
                "images/train/source-a/left_panel/000.png",
                status="accepted",
                transcription="增益",
                rapid="增益",
                vision="增益 ",
                teacher="减益",
            )
        ],
    )
    (record,) = load_studio_records(tmp_path / "batches")[0]
    assert record.options == ["增益", "减益", OPTION_NONE_CORRECT, OPTION_NOT_VALID]
    assert [engine.engine for engine in record.engines] == ["candidate", "rapidocr", "vision", "teacher"]


def test_truth_kind_mapping() -> None:
    assert truth_kind(_record())[0] == "accept_listed"
    assert truth_kind(_record(truth_transcription="未列出的文本"))[0] == "accept_unlisted"
    assert truth_kind(_record(truth_status="rejected", truth_transcription=None))[0] == "reject"
    assert truth_kind(_record(truth_status=None, truth_transcription=None))[0] == "unreviewed"


def test_route_threshold_and_fixed_options() -> None:
    record = _record()
    assert route(_output(record, 0), 0.9) == ("auto_accept", 0)
    assert route(_output(record, 0, confidence=0.5), 0.9) == ("review", None)
    assert route(_output(record, len(record.options) - 1), 0.9) == ("auto_reject", len(record.options) - 1)
    # The none-correct option always routes to human review, whatever its confidence.
    assert route(_output(record, len(record.options) - 2), 0.9) == ("review", None)
    assert route(_output(record, 0, status="error"), 0.9) == ("review", None)


def test_effective_confidence_temperature_scaling() -> None:
    record = _record()
    output = _output(record, 0, confidence=0.99)
    raw = effective_confidence(output)
    calibrated = effective_confidence(output, temperature=4.0)
    assert raw == pytest.approx(0.99)
    assert calibrated is not None and calibrated < raw
    # A threshold between the calibrated and raw confidence routes on the
    # calibrated value: raw would auto-accept, calibrated goes to review.
    assert route(output, 0.7) == ("auto_accept", 0)
    assert route(output, 0.7, temperature=4.0) == ("review", None)


def test_split_assign_is_source_grouped() -> None:
    first = _record(source_id="same-source")
    second = _record(source_id="same-source", record_id="b1:train:y.png")
    assert split_assign(first, 0.5) == split_assign(second, 0.5)


def test_fit_threshold_picks_lowest_feasible() -> None:
    record_ok = _record()
    record_bad = _record(record_id="b1:train:z.png", truth_status="rejected", truth_transcription=None)
    items = [
        (record_ok, _output(record_ok, 0, confidence=0.99)),
        (record_bad, _output(record_bad, 0, confidence=0.6)),
    ]
    threshold, curve = fit_threshold(items, dict(DEFAULT_GATE))
    assert threshold is not None and threshold > 0.6
    feasible = [row for row in curve if row["feasible"]]
    assert feasible and feasible[0]["threshold"] == threshold


def test_run_experiment_mock_runner_and_replay(tmp_path: Path) -> None:
    _batch(
        tmp_path,
        [
            _row(f"images/train/source-{src}/left_panel/{i:03d}.png", source=f"source-{src}", status="accepted", transcription="增益")
            for src in "abcd"
            for i in range(3)
        ]
        + [_row("images/train/source-e/left_panel/000.png", source="source-e", status="pending")],
    )
    records, crop_paths = load_studio_records(tmp_path / "batches")
    runner = MockRunner(accept_probability=0.5)
    result = run_experiment(records, crop_paths, runner, image_tokens_list=[20, 70])
    assert runner.calls == len([r for r in records if r.residual]) * 2
    report = result.metrics
    assert report["inputs"]["residual_rows"] == 13
    assert report["inputs"]["scored_rows"] == 12
    assert set(report["budgets"]) == {"20", "70"}
    assert report["budgets"]["70"]["threshold"] is not None
    assert report["recommendation"]["outcome"] == "keep_sidecar"

    decisions = tmp_path / "decisions.jsonl"
    write_decisions(decisions, result.outputs)
    replayed = load_decisions(decisions)
    runner2 = MockRunner()
    result2 = run_experiment(records, crop_paths, runner2, image_tokens_list=[20, 70], replay_outputs=replayed)
    assert runner2.calls == 0
    assert result2.metrics["budgets"]["70"]["validation"] == report["budgets"]["70"]["validation"]

    write_report(tmp_path / "report", result.metrics)
    assert json.loads((tmp_path / "report" / "report.json").read_text())["issue"] == "ocrkit#25"


def test_records_roundtrip_and_digest_verification(tmp_path: Path) -> None:
    _batch(tmp_path, [_row("images/train/source-a/left_panel/000.png", status="accepted", transcription="增益")])
    records, _ = load_studio_records(tmp_path / "batches")
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    (loaded,) = load_records(path)
    assert loaded.input_digest == records[0].input_digest
    tampered = loaded.model_copy(update={"options": ["篡改", OPTION_NONE_CORRECT, OPTION_NOT_VALID]})
    with pytest.raises(ValueError, match="input_digest"):
        PreReviewRecord(**{**tampered.model_dump(), "input_digest": records[0].input_digest})


def test_missing_crop_fails_clearly(tmp_path: Path) -> None:
    batch = _batch(tmp_path, [_row("images/train/source-a/left_panel/000.png", status="accepted", transcription="增益")])
    (batch / "dataset" / "images/train/source-a/left_panel/000.png").unlink()
    with pytest.raises(ValueError, match="missing crops"):
        load_studio_records(tmp_path / "batches")


def test_omni_worker_protocol_with_stub_model(tmp_path: Path) -> None:
    """The worker speaks JSONL over stdio without importing OCRKit or mlx."""
    model_dir = tmp_path / "stub-model"
    (model_dir / "omni_mlx").mkdir(parents=True)
    (model_dir / "omni_mlx" / "__init__.py").write_text("")
    (model_dir / "omni_mlx" / "classifier.py").write_text(
        "class Classifier:\n"
        "    def __init__(self, path, calibration=None):\n"
        "        self.calibration = None\n"
        "    def predict(self, state, question, options, image=None, image_tokens=70):\n"
        "        probabilities = {option: 0.01 for option in options}\n"
        "        probabilities[options[0]] = 0.97\n"
        "        return {'model': 'stub', 'prediction': options[0], 'prediction_index': 0,\n"
        "                'probabilities': probabilities, 'metrics': {'elapsed_ms': 1.0, 'input_tokens': 3}}\n"
    )
    (model_dir / "conversion.json").write_text(json.dumps({"source": "stub/model", "revision": "abc", "bits": 4}))
    process = subprocess.run(
        [sys.executable, "-u", str(WORKER), "--model-dir", str(model_dir)],
        input=json.dumps({"task_id": "t1", "image": "/x.png", "state": "s", "question": "q", "options": ["a", "b"], "image_tokens": 20}) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert lines[0]["ready"] is True and lines[0]["model"] == "stub/model"
    assert lines[1]["ok"] is True and lines[1]["result"]["prediction_index"] == 0


def test_omni_mlx_runner_against_stub_worker(tmp_path: Path) -> None:
    model_dir = tmp_path / "stub-model"
    (model_dir / "omni_mlx").mkdir(parents=True)
    (model_dir / "omni_mlx" / "__init__.py").write_text("")
    (model_dir / "omni_mlx" / "classifier.py").write_text(
        "class Classifier:\n"
        "    def __init__(self, path, calibration=None):\n"
        "        self.calibration = None\n"
        "    def predict(self, state, question, options, image=None, image_tokens=70):\n"
        "        return {'model': 'stub', 'prediction': options[-2], 'prediction_index': len(options) - 2,\n"
        "                'probabilities': {option: (0.9 if option == options[-2] else 0.1) for option in options},\n"
        "                'metrics': {'elapsed_ms': 1.0}}\n"
    )
    (model_dir / "conversion.json").write_text(json.dumps({"source": "stub/model", "revision": "abc", "bits": 4}))
    runner = OmniMlxRunner(model_dir, Path(sys.executable))
    try:
        output = runner.decide(_record(), Path("unused.png"), 70)
    finally:
        runner.close()
    assert output.status == "ok"
    assert output.selected_option == OPTION_NONE_CORRECT
    assert output.model == "stub/model"
