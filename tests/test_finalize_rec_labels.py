from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.scripts.finalize_rec_labels import finalize


def _write_review(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_finalize_writes_only_reviewed_accepted_labels(tmp_path: Path) -> None:
    _write_review(
        tmp_path / "review/train.jsonl",
        [
            {"crop": "images/train/a.png", "review_status": "accepted", "transcription": "文字"},
            {"crop": "images/train/b.png", "review_status": "rejected", "transcription": None},
        ],
    )
    _write_review(
        tmp_path / "review/holdout.jsonl",
        [{"crop": "images/holdout/c.png", "review_status": "accepted", "transcription": "测试"}],
    )

    assert finalize(tmp_path) == {"train_labels": 1, "holdout_labels": 1}
    assert (tmp_path / "labels/train.txt").read_text(encoding="utf-8") == "images/train/a.png\t文字\n"


def test_finalize_rejects_pending_rows(tmp_path: Path) -> None:
    _write_review(tmp_path / "review/train.jsonl", [{"review_status": "pending"}])
    _write_review(tmp_path / "review/holdout.jsonl", [{"review_status": "rejected"}])

    with pytest.raises(ValueError, match="accepted or rejected"):
        finalize(tmp_path)
