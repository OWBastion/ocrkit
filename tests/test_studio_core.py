from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from training.studio import core as studio_core
from training.studio.core import append_sources, create_batch, export_dataset, generate_candidates, review_counts, review_rows, update_review_row


def _image(path: Path, value: int = 200) -> None:
    encoded, data = cv2.imencode(".png", np.full((40, 60, 3), value, dtype=np.uint8))
    assert encoded
    path.write_bytes(data.tobytes())


def test_create_batch_deduplicates_and_splits_whole_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first)
    _image(second)
    batch_dir, summary = create_batch([first, first, second], work_root=tmp_path / "studio", holdout_ratio=0.5)

    assert summary["sources"] == 1  # identical image bytes are intentionally deduplicated.
    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["split"] == "train"


def test_append_sources_keeps_existing_splits_and_assigns_new_holdout(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first, 100)
    _image(second, 200)
    batch_dir, _ = create_batch([first], work_root=tmp_path / "studio", holdout_ratio=0.2)
    original = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))["sources"][0]

    result = append_sources(batch_dir, [second])

    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert result["added"] == 1
    assert manifest["sources"][0]["id"] == original["id"]
    assert manifest["sources"][0]["split"] == "train"
    assert manifest["sources"][1]["split"] == "holdout"


def test_export_dataset_copies_finalized_batch_into_private_dataset_root(tmp_path: Path, monkeypatch) -> None:
    batch = tmp_path / "batch"
    dataset = batch / "dataset"
    (dataset / "labels").mkdir(parents=True)
    (dataset / "labels/train.txt").write_text("images/train/a.png\t文字\n", encoding="utf-8")
    (dataset / "labels/holdout.txt").write_text("images/holdout/b.png\t文字\n", encoding="utf-8")
    (batch / "batch.json").write_text(json.dumps({"batch_id": "batch-1"}), encoding="utf-8")
    monkeypatch.setattr(studio_core, "finalize_dataset", lambda _: {"validated_train": 1, "validated_holdout": 1})

    result = export_dataset(batch, destination_root=tmp_path / "datasets/labeled/rec/studio")

    assert result["validated_train"] == 1
    assert (tmp_path / "datasets/labeled/rec/studio/batch-1/dataset/labels/train.txt").is_file()


def test_review_updates_are_atomic_and_counted(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    row = {"crop": "images/train/source/000.png", "review_status": "pending", "candidate_text": "候选"}
    (review / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    saved = update_review_row(batch, "train", row["crop"], "accepted", "人工文本")

    assert saved["transcription"] == "人工文本"
    assert review_rows(batch, "train")[0]["review_status"] == "accepted"
    assert review_counts(batch) == {"total": 1, "accepted": 1, "pending": 0, "rejected": 0}


def test_generate_candidates_reuses_completed_review_manifest(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"sources": [{"id": "source-a"}]}), encoding="utf-8")
    row = {"crop": "images/train/source-a/000.png", "source_id": "source-a", "review_status": "accepted", "auto_accept_reason": "rapidocr_vision_agreement"}
    (review / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    summary = generate_candidates(batch)

    assert summary["train_candidates"] == 1
    assert summary["reused_existing_candidates"] is True


def test_generate_candidates_uses_rust_crops_for_new_batch(tmp_path: Path, monkeypatch) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "cases.json").write_text("[]", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_prepare(*args, **kwargs):
        captured.update(kwargs)
        return {
            "cases": 0,
            "train_cases": 0,
            "holdout_cases": 0,
            "train_candidates": 0,
            "holdout_candidates": 0,
            "auto_accepted": 0,
        }

    monkeypatch.setattr(studio_core, "prepare_candidates", fake_prepare)
    generate_candidates(batch)

    assert captured["crop_backend"] == "rust"
