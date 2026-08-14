from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from training.studio import core as studio_core
from training.studio.core import accept_teacher_suggestions, append_sources, create_batch, export_dataset, generate_candidates, refresh_teacher_candidates, refresh_vision_candidates, review_counts, review_rows, update_review_row
from training.vision import VisionLine


def _image(path: Path, value: int = 200) -> None:
    encoded, data = cv2.imencode(".png", np.full((40, 60, 3), value, dtype=np.uint8))
    assert encoded
    path.write_bytes(data.tobytes())


def _sized_image(path: Path, width: int, height: int, value: int = 200) -> None:
    encoded, data = cv2.imencode(".png", np.full((height, width, 3), value, dtype=np.uint8))
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


def test_create_batch_selects_roi_layout_per_source_aspect_ratio(tmp_path: Path) -> None:
    tall = tmp_path / "tall.png"
    wide = tmp_path / "wide.png"
    _sized_image(tall, 2560, 1600, 100)
    _sized_image(wide, 1920, 1080, 200)

    batch_dir, summary = create_batch([tall, wide], work_root=tmp_path / "studio", holdout_ratio=0.5)

    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    by_name = {row["original_name"]: row for row in manifest["sources"]}
    assert summary["layout_version"] == "mixed"
    assert by_name["tall.png"]["layout_version"] == "1280x800-v1"
    assert by_name["wide.png"]["layout_version"] == "1280x720-v6"
    assert by_name["tall.png"]["quality"]["warnings"] == []
    assert by_name["wide.png"]["quality"]["warnings"] == []


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
    assert review_counts(batch) == {"total": 1, "accepted": 1, "pending": 0, "rejected": 0, "teacher_eligible": 0}


def test_manual_rejection_is_saved_to_negative_registry(tmp_path: Path) -> None:
    batch = tmp_path / "studio/batches/batch-1"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    row = {
        "crop": "images/train/source/000.png",
        "roi": "left_panel",
        "candidate_text": "错误内容",
        "rapidocr_text": "错误内容",
        "review_status": "pending",
    }
    (review / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    update_review_row(batch, "train", row["crop"], "rejected", None)

    registry = tmp_path / "studio/negative-candidates.jsonl"
    entries = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries == [{
        "schema_version": 1,
        "roi": "left_panel",
        "texts": ["错误内容"],
        "crop_sha256": None,
        "raw_roi_sha256": None,
    }]


def test_auto_accepted_rows_are_filterable_and_manual_text_override_clears_marker(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    row = {
        "crop": "images/train/source/000.png",
        "review_status": "accepted",
        "transcription": "模型文本",
        "auto_accept_reason": "rapidocr_vision_agreement",
    }
    (review / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    assert review_rows(batch, "train", "auto_accepted")[0]["crop"] == row["crop"]
    updated = update_review_row(batch, "train", row["crop"], "accepted", "人工修正")

    assert updated["transcription"] == "人工修正"
    assert updated["auto_accept_reason"] is None


def test_accept_teacher_suggestions_only_accepts_eligible_train_rows(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    rows = [
        {
            "crop": "images/train/source/000.png",
            "review_status": "pending",
            "teacher_text": "教师文本",
            "teacher_auto_accept_eligible": True,
        },
        {
            "crop": "images/train/source/001.png",
            "review_status": "pending",
            "teacher_text": "不应自动接受",
            "teacher_auto_accept_eligible": False,
        },
    ]
    (review / "train.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    result = accept_teacher_suggestions(batch)
    saved = review_rows(batch, "train")

    assert result == {"accepted": 1, "pending": 1, "teacher_eligible": 0}
    assert saved[0]["review_status"] == "accepted"
    assert saved[0]["transcription"] == "教师文本"
    assert saved[0]["auto_accept_reason"] == "teacher_model_agreement"
    assert saved[1]["review_status"] == "pending"


def test_refresh_teacher_preserves_manual_decisions_and_adds_predictions(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    dataset = batch / "dataset"
    review = dataset / "review"
    (dataset / "images/train/source-a").mkdir(parents=True)
    review.mkdir(parents=True)
    _image(dataset / "images/train/source-a/000.png")
    rows = [
        {"crop": "images/train/source-a/000.png", "candidate_text": "教师文本", "rapidocr_text": "教师文本", "rapidocr_confidence": 0.99, "review_status": "pending"},
        {"crop": "images/train/source-a/000.png", "candidate_text": "教师文本", "review_status": "accepted", "transcription": "人工文本"},
        {"crop": "images/train/source-a/000.png", "candidate_text": "教师文本", "review_status": "rejected", "transcription": None},
    ]
    (review / "train.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    class FakeTeacher:
        def __call__(self, image: np.ndarray, *, use_det: bool, use_cls: bool) -> SimpleNamespace:
            assert use_det is False and use_cls is False
            return SimpleNamespace(txts=("教师文本",), scores=(0.99,))

    summary = refresh_teacher_candidates(batch, teacher_factory=FakeTeacher, teacher_model_version="v1")
    saved = review_rows(batch, "train")

    assert summary["rows"] == 3
    assert summary["teacher_covered"] == 3
    assert summary["teacher_auto_accepted"] == 1
    assert summary["preserved_accepted"] == 1
    assert summary["preserved_rejected"] == 1
    assert saved[0]["teacher_text"] == "教师文本"
    assert saved[0]["review_status"] == "accepted"
    assert saved[0]["auto_accept_reason"] == "teacher_rapidocr_agreement"
    assert saved[1]["transcription"] == "人工文本"
    assert saved[2]["review_status"] == "rejected"


def test_refresh_teacher_auto_rejects_wrong_content_in_run_code_roi(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    dataset = batch / "dataset"
    review = dataset / "review"
    (dataset / "images/train/source-a").mkdir(parents=True)
    review.mkdir(parents=True)
    _image(dataset / "images/train/source-a/000.png")
    row = {
        "crop": "images/train/source-a/000.png",
        "roi": "run_code_panel",
        "rapidocr_text": "保持距离(31秒)",
        "rapidocr_confidence": 0.99,
        "review_status": "pending",
    }
    (review / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    class FakeTeacher:
        def __call__(self, image: np.ndarray, *, use_det: bool, use_cls: bool) -> SimpleNamespace:
            assert use_det is False and use_cls is False
            return SimpleNamespace(txts=("保持距离(31秒)",), scores=(0.99,))

    summary = refresh_teacher_candidates(batch, teacher_factory=FakeTeacher, teacher_model_version="v1")
    saved = review_rows(batch, "train")[0]

    assert summary["auto_rejected"] == 1
    assert saved["review_status"] == "rejected"
    assert saved["auto_reject_reason"] == "run_code.content_mismatch"


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


def test_generate_candidates_applies_human_negative_memory_to_pending_rows(tmp_path: Path) -> None:
    work_root = tmp_path / "studio"
    prior = work_root / "batches/prior"
    current = work_root / "batches/current"
    prior_review = prior / "dataset/review"
    current_review = current / "dataset/review"
    prior_review.mkdir(parents=True)
    (current / "batch.json").parent.mkdir(parents=True, exist_ok=True)
    current_review.mkdir(parents=True)
    (prior / "batch.json").write_text(json.dumps({"sources": []}), encoding="utf-8")
    (current / "batch.json").write_text(json.dumps({"sources": [{"id": "source-current"}]}), encoding="utf-8")
    rejected = {
        "roi": "left_panel",
        "candidate_text": "错误内容",
        "rapidocr_text": "错误内容",
        "review_status": "rejected",
        "auto_reject_reason": None,
    }
    pending = {
        "crop": "images/train/source-current/000.png",
        "source_id": "source-current",
        "roi": "left_panel",
        "candidate_text": "错误内容",
        "rapidocr_text": "错误内容",
        "review_status": "pending",
    }
    (prior_review / "train.jsonl").write_text(json.dumps(rejected, ensure_ascii=False) + "\n", encoding="utf-8")
    (prior_review / "holdout.jsonl").write_text("", encoding="utf-8")
    (current_review / "train.jsonl").write_text(json.dumps(pending, ensure_ascii=False) + "\n", encoding="utf-8")
    (current_review / "holdout.jsonl").write_text("", encoding="utf-8")
    crop_path = current / "dataset/images/train/source-current/000.png"
    crop_path.parent.mkdir(parents=True)
    _image(crop_path)

    summary = generate_candidates(current)
    saved = review_rows(current, "train")[0]

    assert summary["negative_auto_rejected"] == 1
    assert saved["review_status"] == "rejected"
    assert saved["auto_reject_reason"] == "negative_review.text_match"


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


def test_refresh_vision_preserves_manual_review_and_updates_pending_rows(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    dataset = batch / "dataset"
    review = dataset / "review"
    (dataset / "images/train/source-a").mkdir(parents=True)
    review.mkdir(parents=True)
    image = cv2.imencode(".png", np.full((40, 60, 3), 200, dtype=np.uint8))[1].tobytes()
    (dataset / "images/train/source-a/000.png").write_bytes(image)
    pending = {
        "crop": "images/train/source-a/000.png",
        "rapidocr_text": "模型文本",
        "rapidocr_confidence": 0.99,
        "review_status": "pending",
        "transcription": None,
    }
    manual = {
        "crop": "images/train/source-a/000.png",
        "rapidocr_text": "模型文本",
        "rapidocr_confidence": 0.99,
        "review_status": "accepted",
        "transcription": "人工文本",
        "auto_accept_reason": None,
    }
    rejected = {
        "crop": "images/train/source-a/000.png",
        "rapidocr_text": "模型文本",
        "rapidocr_confidence": 0.99,
        "review_status": "rejected",
        "transcription": None,
        "auto_accept_reason": None,
    }
    (review / "train.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (pending, manual, rejected)) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")

    class FakeVision:
        def recognize(self, _image: np.ndarray) -> list[VisionLine]:
            return [VisionLine("模型文本", 0.99, np.zeros((4, 2), dtype=np.float32))]

    summary = refresh_vision_candidates(batch, vision_factory=FakeVision)
    rows = review_rows(batch, "train")

    assert summary == {"rows": 3, "vision_covered": 3, "auto_accepted": 1, "auto_rejected": 0, "preserved_accepted": 1, "preserved_rejected": 1}
    assert rows[0]["review_status"] == "accepted"
    assert rows[0]["transcription"] == "模型文本"
    assert rows[1]["transcription"] == "人工文本"
    assert rows[2]["review_status"] == "rejected"
