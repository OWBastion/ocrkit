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


def test_evaluate_excludes_rapidocr_vision_agreement_when_previous_model_disagrees(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    review = tmp_path / "review"
    labels.mkdir()
    review.mkdir()
    (labels / "holdout.txt").write_text("images/holdout/a.png\t挑战 完成\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text(
        json.dumps(
            {
                "crop": "images/holdout/a.png",
                "rapidocr_text": "挑战 完成",
                "vision_text": "挑战 完成",
                "teacher_text": "挑战 失败",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert evaluate(tmp_path)["agreement"] == {"covered": 0, "correct": 0}


def test_evaluate_reports_raw_and_normalized_accuracy_separately(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    review = tmp_path / "review"
    labels.mkdir()
    review.mkdir()
    (labels / "holdout.txt").write_text(
        "images/holdout/a.png\t增益\nimages/holdout/b.png\t减益\n", encoding="utf-8"
    )
    rows = [
        {
            "crop": "images/holdout/a.png",
            "layout_version": "1280x720-v6",
            "roi": "left_panel",
            "rapidocr_text": "编益",
        },
        {
            "crop": "images/holdout/b.png",
            "layout_version": "1280x720-v6",
            "roi": "left_panel",
            "rapidocr_text": "减益",
        },
    ]
    (review / "holdout.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    from app.parser.terminology import load_terminology_rules

    result = evaluate(tmp_path, load_terminology_rules(Path("configs/terminology.yaml")))
    terminology = result["terminology"]
    assert terminology["rules_version"] == "1"
    # Raw accuracy: 编益 is wrong, 减益 is right.
    assert terminology["raw"]["rapidocr"] == {"covered": 2, "correct": 1}
    # Normalized accuracy: both are right after normalization.
    assert terminology["normalized"]["rapidocr"] == {"covered": 2, "correct": 2}
    assert terminology["adopted"] == 1
    assert terminology["corrected"] == 1
    assert terminology["false_corrections"] == 0
    assert terminology["hit_rate"] == 1.0
    assert terminology["false_correction_rate"] == 0.0
    assert terminology["decisions"]["normalized"] == 1
    assert terminology["decisions"]["unchanged"] == 1
