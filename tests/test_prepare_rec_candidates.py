from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
from training.scripts.prepare_rec_candidates import HOLDOUT_IDS, prepare_candidates, split_for_case
from training.vision import VisionLine


class FakeResult:
    boxes = np.array([[[2, 3], [12, 3], [12, 9], [2, 9]]], dtype=np.float32)
    txts = ("候选文字",)
    scores = (0.98765,)


class FakeRapidOCR:
    def __call__(self, image: np.ndarray, use_cls: bool) -> FakeResult:
        assert use_cls is False
        return FakeResult()


class FakeVisionOcr:
    def recognize(self, image: np.ndarray) -> list[VisionLine]:
        return [VisionLine("另一候选", 0.99, FakeResult.boxes[0])]


def test_prepare_candidates_creates_review_and_empty_label_scaffolds(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    assert cv2.imencode(".png", source)[0]
    (fixtures / "sample.png").write_bytes(cv2.imencode(".png", source)[1].tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "sample_01", "image": "sample.png"}], ensure_ascii=False), encoding="utf-8"
    )
    config = RoiConfig(width=60, height=40, rois={"left_panel": RoiBox(0, 0, 30, 20)})

    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=FakeRapidOCR,
        vision_factory=FakeVisionOcr,
    )

    assert summary == {
        "cases": 1,
        "train_cases": 1,
        "holdout_cases": 0,
        "train_candidates": 1,
        "holdout_candidates": 0,
        "auto_accepted": 0,
    }
    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert row["candidate_text"] == "候选文字"
    assert row["review_status"] == "pending"
    crop = cv2.imread(str(tmp_path / "labeled" / row["crop"]))
    assert crop is not None and crop.shape[2] == 3
    assert (tmp_path / "labeled/labels/train.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "labeled/labels/holdout.txt").read_text(encoding="utf-8") == ""


def test_holdout_split_is_fixed_to_training_plan() -> None:
    assert split_for_case("samoa_hell_01") == "holdout"
    assert split_for_case("route_66_01") == "holdout"
    assert split_for_case("not_in_plan") == "train"
    assert len(HOLDOUT_IDS) == 8
