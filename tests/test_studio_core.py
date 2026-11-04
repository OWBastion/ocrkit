from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from training.studio.core import accept_teacher_suggestions, append_sources, create_batch, review_rows


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
