from __future__ import annotations

import json
from pathlib import Path

from training.scripts.evaluate_rec_candidates import evaluate


def test_evaluate_compares_each_engine_and_agreement_to_holdout_truth(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    review = tmp_path / "review"
    labels.mkdir()
    review.mkdir()
    (labels / "holdout.txt").write_text(
        "images/holdout/a.png\t挑战 完成\nimages/holdout/b.png\t66号公路\n", encoding="utf-8"
    )
    rows = [
        {"crop": "images/holdout/a.png", "rapidocr_text": "挑战  完成", "vision_text": "挑战 完成"},
        {"crop": "images/holdout/b.png", "rapidocr_text": "66号公路", "vision_text": "错误"},
    ]
    (review / "holdout.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    assert evaluate(tmp_path) == {
        "labels": {"total": 2},
        "rapidocr": {"covered": 2, "correct": 2},
        "vision": {"covered": 2, "correct": 1},
        "teacher": {"covered": 0, "correct": 0},
        "agreement": {"covered": 1, "correct": 1},
    }
