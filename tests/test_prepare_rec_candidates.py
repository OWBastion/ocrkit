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


def test_prepare_candidates_auto_rejects_text_that_does_not_match_run_code_roi(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    encoded, data = cv2.imencode(".png", source)
    assert encoded
    (fixtures / "sample.png").write_bytes(data.tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "sample_01", "image": "sample.png"}], ensure_ascii=False), encoding="utf-8"
    )
    config = RoiConfig(width=60, height=40, rois={"run_code_panel": RoiBox(0, 0, 30, 20)})

    class WrongRapidOCR:
        def __call__(self, image: np.ndarray, use_cls: bool) -> object:
            assert use_cls is False
            return type(
                "WrongResult", (), {"boxes": FakeResult.boxes, "txts": ("保持距离(31秒)",), "scores": (0.99,)}
            )()

    class WrongVisionOcr:
        def recognize(self, image: np.ndarray) -> list[VisionLine]:
            return [VisionLine("保持距离(31秒)", 0.99, FakeResult.boxes[0])]

    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=WrongRapidOCR,
        vision_factory=WrongVisionOcr,
    )

    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert summary["auto_rejected"] == 1
    assert row["review_status"] == "rejected"
    assert row["auto_reject_reason"] == "run_code.content_mismatch"
    assert row["transcription"] is None


def test_prepare_candidates_does_not_reject_repeated_text_without_same_crop(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    encoded, data = cv2.imencode(".png", source)
    assert encoded
    (fixtures / "sample.png").write_bytes(data.tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "sample_01", "image": "sample.png"}], ensure_ascii=False), encoding="utf-8"
    )
    negative_path = tmp_path / "negative-candidates.jsonl"
    negative_path.write_text(
        json.dumps({"roi": "left_panel", "texts": ["候选文字"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = RoiConfig(width=60, height=40, rois={"left_panel": RoiBox(0, 0, 30, 20)})

    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=FakeRapidOCR,
        vision_factory=FakeVisionOcr,
        negative_examples_path=negative_path,
    )

    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert summary["auto_rejected"] == 0
    assert row["review_status"] == "pending"
    assert row["auto_reject_reason"] is None


def test_holdout_split_is_fixed_to_training_plan() -> None:
    assert split_for_case("samoa_hell_01") == "holdout"
    assert split_for_case("route_66_01") == "holdout"
    assert split_for_case("not_in_plan") == "train"
    assert len(HOLDOUT_IDS) == 8


def test_prepare_candidates_honors_explicit_source_level_split(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    (fixtures / "sample.png").write_bytes(cv2.imencode(".png", source)[1].tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "not_in_fixed_holdout", "image": "sample.png", "split": "holdout"}]), encoding="utf-8"
    )
    config = RoiConfig(width=60, height=40, rois={"left_panel": RoiBox(0, 0, 30, 20)})

    summary = prepare_candidates(
        fixtures / "cases.json", fixtures, tmp_path / "labeled", config, ocr_factory=FakeRapidOCR, vision_factory=FakeVisionOcr
    )

    assert summary["train_cases"] == 0
    assert summary["holdout_cases"] == 1
    rows = [json.loads(line) for line in (tmp_path / "labeled/review/holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["split"] == "holdout"


def test_known_irrelevant_text_is_rejected_by_roi_but_real_fields_are_not() -> None:
    from training.scripts.prepare_rec_candidates import candidate_rejection_reason

    assert candidate_rejection_reason("bottom_left_hero", ["631 | 631", "631/631"]) == "bottom_left_hero.not_player_name"
    assert candidate_rejection_reason("bottom_left_hero", ["八月八月八"]) is None
    for overlay in ("FPS: 144.", "VRM: 4972 MB", "延迟: 45 MS", "IND: 32 MS", "12:37 AM", "C"):
        assert candidate_rejection_reason("right_panel", [overlay]) == "right_panel.performance_overlay"
    for field in ("皇家赛道：地狱", "[随机事件5.0]", "版本26.0926.1", "服务器负载147", "已运行3小时21分22秒"):
        assert candidate_rejection_reason("right_panel", [field]) is None
    # The run code row has ROIs of its own; stats lines and other ROIs are not touched.
    assert candidate_rejection_reason("right_panel", ["3690-4655-6310"]) == "right_panel.run_code_row"
    assert candidate_rejection_reason("left_panel", ["3690-4655-6310"]) == "left_panel.run_code_row"
    for stats in ("58/8/32/-972%", "1小时31分34秒", "78/0", "通关总计耗时"):
        assert candidate_rejection_reason("left_panel", [stats]) is None
    assert candidate_rejection_reason("run_code_panel", ["3690-4655-6310"]) is None
    # Only the listed ROIs are filtered.
    assert candidate_rejection_reason("left_panel", ["FPS: 144"]) is None
