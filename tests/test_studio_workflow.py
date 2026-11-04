from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.jev.records import OPTION_NONE_CORRECT, compute_input_digest, record_from_review_row
from training.jev.runner import DecisionOutput
from training.studio import workflow
from training.studio.core import review_rows, save_jev_suggestion


def _batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batches/example"
    (batch / "dataset/review").mkdir(parents=True)
    (batch / "dataset/images").mkdir()
    (batch / "batch.json").write_text(json.dumps({"layout_version": "v1", "sources": []}))
    (batch / "dataset/images/pending.png").write_bytes(b"pending crop")
    (batch / "dataset/images/manual.png").write_bytes(b"reviewed crop")
    rows = [{"crop": "images/pending.png", "source_id": "source1", "roi": "map_panel", "candidate_text": "Samoa", "review_status": "pending"}, {"crop": "images/manual.png", "source_id": "source2", "roi": "map_panel", "candidate_text": "wrong", "review_status": "accepted", "transcription": "human truth", "review_method": "human"}]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text("")
    return batch


def test_suggestions_are_digest_bound_and_do_not_become_labels(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    row = review_rows(batch, "train")[0]
    record = record_from_review_row(batch.name, "train", "v1", hashlib.sha256(b"pending crop").hexdigest(), row)
    suggestion = {"input_digest": compute_input_digest(record), "action": "accept", "selected_option": "Samoa"}
    assert save_jev_suggestion(batch, "train", row["crop"], suggestion)
    current = review_rows(batch, "train")
    assert current[0]["review_status"] == "pending"
    assert "transcription" not in current[0]
    assert current[1]["transcription"] == "human truth"
    assert not (batch / "dataset/labels").exists()
    (batch / "dataset/images/pending.png").write_bytes(b"changed image")
    assert not save_jev_suggestion(batch, "train", row["crop"], suggestion)
    assert not save_jev_suggestion(batch, "train", "images/manual.png", suggestion)


def test_worker_cannot_overwrite_a_human_decision_made_during_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    class Runner:
        def __init__(self, *args: object) -> None:
            pass
        def decide(self, record, image: Path, tokens: int) -> DecisionOutput:
            from training.studio.core import update_review_row
            update_review_row(batch, "train", record.crop, "accepted", "human correction")
            return DecisionOutput(record_id=record.record_id, input_digest=compute_input_digest(record), runner="test", status="ok", reason_code="ok", image_tokens=tokens, options=record.options, selected_option="Samoa")
        def close(self) -> None:
            pass
    monkeypatch.setattr(workflow, "OmniMlxRunner", Runner)
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True, "model_dir": "local", "worker_python": "python"})
    workflow.run_jev(batch, run, 70)
    row = review_rows(batch, "train")[0]
    assert row["transcription"] == "human correction"
    assert "jev_suggestion" not in row
    assert json.loads((run / "result.json").read_text())["saved"] == 0


def _suggest(batch: Path, split: str, crop: str, **suggestion: object) -> None:
    path = batch / "dataset/review" / f"{split}.jsonl"
    rows = review_rows(batch, split)
    next(row for row in rows if row["crop"] == crop)["jev_suggestion"] = {"status": "ok", **suggestion}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_confident_jev_choices_are_adopted_for_train_only(tmp_path: Path) -> None:
    from training.studio.core import adopt_jev_suggestions

    batch = _batch(tmp_path)
    rows = review_rows(batch, "train")
    rows += [
        {"crop": f"images/{name}.png", "source_id": name, "roi": "map_panel", "review_status": "pending"}
        for name in ("low", "reject")
    ]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text(json.dumps({"crop": "images/h.png", "source_id": "h", "roi": "map_panel", "review_status": "pending", "jev_suggestion": {"status": "ok", "action": "accept", "selected_option": "Samoa", "confidence": 0.99}}) + "\n")
    _suggest(batch, "train", "images/pending.png", action="accept", selected_option="Samoa", confidence=0.95)
    _suggest(batch, "train", "images/manual.png", action="accept", selected_option="jev guess", confidence=0.99)
    _suggest(batch, "train", "images/low.png", action="accept", selected_option="maybe", confidence=0.5)
    _suggest(batch, "train", "images/reject.png", action="reject", selected_option="not valid", confidence=0.97)
    _suggest(batch, "train", "images/manual.png", action="accept", selected_option="jev guess", confidence=0.99)
    _suggest(batch, "train", "images/manual.png", action="manual", selected_option=None, confidence=0.99)

    assert adopt_jev_suggestions(batch) == {"accepted": 1, "rejected": 1}
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/pending.png"]["review_status"] == "accepted"
    assert by_crop["images/pending.png"]["transcription"] == "Samoa"
    assert by_crop["images/pending.png"]["review_method"] == "automatic"
    assert by_crop["images/reject.png"]["review_status"] == "rejected"
    assert by_crop["images/low.png"]["review_status"] == "pending"
    assert by_crop["images/manual.png"]["transcription"] == "human truth"
    assert review_rows(batch, "holdout")[0]["review_status"] == "pending"


def test_engine_agreement_skips_jev_and_is_adopted_for_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from training.studio.core import adopt_engine_agreement, engines_agree

    batch = _batch(tmp_path)
    agree = {"rapidocr_text": "Samoa ", "vision_text": "Samoa", "review_status": "pending", "roi": "map_panel"}
    rows = [
        {**agree, "crop": "images/agree.png", "source_id": "a"},
        {**agree, "crop": "images/teacher.png", "source_id": "b", "teacher_text": "Other"},
        {**agree, "crop": "images/differ.png", "source_id": "c", "vision_text": "Samba"},
        {**agree, "crop": "images/single.png", "source_id": "d", "vision_text": None},
    ]
    holdout = [{**agree, "crop": "images/h.png", "source_id": "h"}]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in holdout))
    assert [engines_agree(row) for row in rows] == [True, False, False, False]

    seen = []
    class Runner:
        def __init__(self, *args: object) -> None:
            pass
        def decide(self, record, image: Path, tokens: int) -> DecisionOutput:
            seen.append(record.record_id)
            return DecisionOutput(record_id=record.record_id, input_digest=compute_input_digest(record), runner="test", status="ok", reason_code="ok", image_tokens=tokens, options=record.options, selected_option=OPTION_NONE_CORRECT)
        def close(self) -> None:
            pass
    for name in ("agree", "teacher", "differ", "single", "h"):
        (batch / f"dataset/images/{name}.png").write_bytes(name.encode())
    monkeypatch.setattr(workflow, "OmniMlxRunner", Runner)
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True, "model_dir": "local", "worker_python": "python"})
    run = tmp_path / "run"
    run.mkdir()
    workflow.run_jev(batch, run, 70)
    assert len(seen) == 3  # teacher/differ/single only; agreeing Train and Holdout rows skipped

    assert adopt_engine_agreement(batch) == 1
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/agree.png"]["review_status"] == "accepted"
    assert by_crop["images/agree.png"]["transcription"] == "Samoa"
    assert by_crop["images/teacher.png"]["review_status"] == "pending"
    assert review_rows(batch, "holdout")[0]["review_status"] == "pending"


def test_ai_watermark_rejects_the_rest_of_that_screenshot_but_keeps_the_mark_and_human_work(tmp_path: Path) -> None:
    from training.studio.core import reject_ai_generated_sources, update_review_row

    batch = _batch(tmp_path)
    (batch / "batch.json").write_text(json.dumps({"layout_version": "v1", "sources": [{"id": "ai"}, {"id": "real"}]}))
    rows = [
        {"crop": "images/mark.png", "source_id": "ai", "roi": "ai_watermark", "rapidocr_text": "豆包AI生成", "review_status": "pending"},
        {"crop": "images/left.png", "source_id": "ai", "roi": "left_panel", "review_status": "pending"},
        {"crop": "images/done.png", "source_id": "ai", "roi": "left_panel", "review_status": "accepted", "transcription": "kept", "review_method": "human"},
        {"crop": "images/real.png", "source_id": "real", "roi": "left_panel", "review_status": "pending"},
        {"crop": "images/lookalike.png", "source_id": "real", "roi": "ai_watermark", "rapidocr_text": "AI", "review_status": "pending"},
    ]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert reject_ai_generated_sources(batch) == 1
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/left.png"]["review_status"] == "rejected"
    assert by_crop["images/left.png"]["auto_reject_reason"] == "ai_generated"
    assert by_crop["images/mark.png"]["review_status"] == "pending"
    assert by_crop["images/done.png"]["transcription"] == "kept"
    assert by_crop["images/real.png"]["review_status"] == "pending"
    assert [s.get("ai_generated") for s in json.loads((batch / "batch.json").read_text())["sources"]] == [True, None]

    row = update_review_row(batch, "train", "images/left.png", "accepted", "x", "event")
    assert row["category"] == "event"
    with pytest.raises(ValueError, match="category"):
        update_review_row(batch, "train", "images/left.png", "accepted", "x", "nope")


def _kaggle_run(batch: Path, *, passed: bool) -> Path:
    output = batch / "runs/kaggle-20260101-000000-abcd1234/output"
    base = output if passed else output / "partial"
    (base / "checkpoint").mkdir(parents=True)
    (base / "checkpoint/best_accuracy.pdparams").write_bytes(b"weights")
    (base / "run.json").write_text(json.dumps({"status": "success"}))
    (base / "evaluation").mkdir()
    (base / "evaluation/fixture_report.json").write_text(json.dumps({"field_accuracy": 0.92}))
    (output / "status.json").write_text(json.dumps({"status": "failed", "error": "RuntimeError: local checkpoint evaluation failed", "kaggle_kernel": "k"}))
    return output


def test_evaluate_that_fails_the_gate_leaves_the_failed_run_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    output = _kaggle_run(batch, passed=False)
    run = tmp_path / "evaluate-run"
    run.mkdir()
    monkeypatch.setattr(workflow.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=1))

    assert workflow.run_evaluate(batch, run) == 1
    assert (output / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert "checkpoint" not in json.loads((run / "result.json").read_text())


def _finalized_batch(root: Path, batch_id: str, sources: dict[str, tuple[str, str]], rows: dict[str, list[tuple[str, str]]]) -> Path:
    """sources: id -> (sha, split); rows: split -> [(source id, text)]."""
    batch = root / "batches" / batch_id
    (batch / "dataset/labels").mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"batch_id": batch_id, "sources": [{"id": i, "sha256": sha, "split": split} for i, (sha, split) in sources.items()]}))
    for split, items in rows.items():
        lines = []
        for index, (source, text) in enumerate(items):
            relative = f"images/{split}/{source}/left_panel/{index:03d}.png"
            (batch / "dataset" / relative).parent.mkdir(parents=True, exist_ok=True)
            (batch / "dataset" / relative).write_bytes(f"{batch_id}{relative}".encode())
            lines.append(f"{relative}\t{text}\n")
        (batch / "dataset/labels" / f"{split}.txt").write_text("".join(lines))
    return batch


def test_merge_keeps_the_first_batch_first_and_never_trains_on_a_held_out_screenshot(tmp_path: Path) -> None:
    from training.studio.core import merge_batch_datasets, mergeable_batches

    current = _finalized_batch(tmp_path, "b2", {"s1": ("sha-a", "train"), "s2": ("sha-b", "holdout")}, {"train": [("s1", "甲")], "holdout": [("s2", "乙")]})
    older = _finalized_batch(
        tmp_path, "b1",
        {"s1": ("sha-b", "train"), "s2": ("sha-a", "train"), "s3": ("sha-c", "holdout")},
        {"train": [("s1", "held out in b2"), ("s2", "duplicate of b2 train")], "holdout": [("s3", "丙")]},
    )

    merged = merge_batch_datasets([current, older], tmp_path / "merged")

    train = (tmp_path / "merged/labels/train.txt").read_text().splitlines()
    holdout = (tmp_path / "merged/labels/holdout.txt").read_text().splitlines()
    assert [line.split("\t")[1] for line in train] == ["甲"]
    assert sorted(line.split("\t")[1] for line in holdout) == ["丙", "乙"]
    assert all(line.startswith("images/b2/") for line in train)
    assert all((tmp_path / "merged" / line.split("\t")[0]).is_file() for line in train + holdout)
    older_counts = next(b for b in merged["batches"] if b["batch_id"] == "b1")
    assert older_counts["dropped_held_out_elsewhere"] == 1
    assert older_counts["dropped_duplicate_source"] == 1
    assert (merged["batch_id"], merged["train_samples"], merged["holdout_samples"]) == ("b2", 1, 2)
    assert json.loads((tmp_path / "merged/merge_manifest.json").read_text())["train_samples"] == 1
    assert [b["batch_id"] for b in mergeable_batches(tmp_path, exclude="b2")] == ["b1"]


def test_only_a_failed_comparison_can_be_overridden(tmp_path: Path) -> None:
    from training.scripts.accept_comparison_override import accept

    (tmp_path / "comparison_report.json").write_text(json.dumps({"verdict": "passed"}))
    with pytest.raises(ValueError, match="failed comparison"):
        accept(tmp_path, "there is nothing to override here")
