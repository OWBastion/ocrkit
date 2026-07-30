from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
import training.scripts.prepare_rec_candidates as prepare_module
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


def test_prepare_candidates_can_consume_rust_crop_manifest(tmp_path: Path, monkeypatch) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    (fixtures / "sample.png").write_bytes(cv2.imencode(".png", source)[1].tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "sample_01", "image": "sample.png"}]), encoding="utf-8"
    )
    config = RoiConfig(width=60, height=40, rois={"left_panel": RoiBox(0, 0, 30, 20)})

    def fake_rust_crops(cases_path: Path, fixture_dir: Path, roi_config_path: Path, workspace: Path) -> dict:
        crop_root = workspace / "rust-crops/images/train/sample_01/left_panel"
        crop_root.mkdir(parents=True)
        crop_path = crop_root / "000.png"
        crop_path.write_bytes(cv2.imencode(".png", np.full((20, 30, 3), 200, dtype=np.uint8))[1].tobytes())
        (workspace / "rust-crops/crop_manifest.json").write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "sample_01",
                            "rois": {
                                "left_panel": {
                                    "path": "images/train/sample_01/left_panel/000.png",
                                    "sha256": "raw-roi-hash",
                                }
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {("sample_01", "left_panel"): {"path": str(crop_path), "sha256": "raw-roi-hash"}}

    monkeypatch.setattr(prepare_module, "_run_rust_crop_batch", fake_rust_crops)
    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=FakeRapidOCR,
        vision_factory=FakeVisionOcr,
        crop_backend="rust",
    )

    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert summary["train_candidates"] == 1
    assert row["crop_backend"] == "rust"
    assert row["raw_roi_sha256"] == "raw-roi-hash"
    assert (tmp_path / "labeled/crop_manifest.json").is_file()
