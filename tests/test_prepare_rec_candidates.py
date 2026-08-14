from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
import training.scripts.prepare_rec_candidates as prepare_module
from training.scripts.prepare_rec_candidates import HOLDOUT_IDS, discover_candidate_artifact, prepare_candidates, split_for_case
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
        "auto_rejected": 0,
        "deduplicated": 0,
        "teacher_auto_accepted": 0,
        "teacher_model_version": None,
        "teacher_suggestions": 0,
        "teacher_auto_accept_eligible": 0,
    }
    row = json.loads((tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8"))
    assert row["candidate_text"] == "候选文字"
    assert row["review_status"] == "pending"
    crop = cv2.imread(str(tmp_path / "labeled" / row["crop"]))
    assert crop is not None and crop.shape[2] == 3
    assert (tmp_path / "labeled/labels/train.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "labeled/labels/holdout.txt").read_text(encoding="utf-8") == ""


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


def test_prepare_candidates_auto_rejects_a_prior_human_negative_for_any_roi(tmp_path: Path) -> None:
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
    assert summary["auto_rejected"] == 1
    assert row["review_status"] == "rejected"
    assert row["auto_reject_reason"] == "negative_review.text_match"


def test_prepare_candidates_deduplicates_overlapping_achievement_and_left_panel_rows(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    source = np.full((40, 60, 3), 200, dtype=np.uint8)
    encoded, data = cv2.imencode(".png", source)
    assert encoded
    (fixtures / "sample.png").write_bytes(data.tobytes())
    (fixtures / "cases.json").write_text(
        json.dumps([{"id": "sample_01", "image": "sample.png"}], ensure_ascii=False), encoding="utf-8"
    )
    config = RoiConfig(
        width=60,
        height=40,
        rois={
            "left_panel": RoiBox(0, 0, 30, 20),
            "achievement_panel": RoiBox(0, 0, 30, 20),
        },
    )

    summary = prepare_candidates(
        fixtures / "cases.json",
        fixtures,
        tmp_path / "labeled",
        config,
        ocr_factory=FakeRapidOCR,
        vision_factory=FakeVisionOcr,
    )

    rows = [json.loads(line) for line in (tmp_path / "labeled/review/train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["deduplicated"] == 1
    assert {row["roi"] for row in rows} == {"left_panel", "achievement_panel"}
    assert next(row for row in rows if row["roi"] == "achievement_panel")["review_status"] == "pending"
    left = next(row for row in rows if row["roi"] == "left_panel")
    assert left["review_status"] == "rejected"
    assert left["auto_reject_reason"] == "duplicate_roi_candidate"


def test_holdout_split_is_fixed_to_training_plan() -> None:
    assert split_for_case("samoa_hell_01") == "holdout"
    assert split_for_case("route_66_01") == "holdout"
    assert split_for_case("not_in_plan") == "train"
    assert len(HOLDOUT_IDS) == 8


def test_discover_candidate_artifact_ignores_unmanifested_fixture_models(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture-rec-99999999"
    fixture.mkdir()
    for name in ("rapidocr.yaml", "det.onnx", "rec.onnx", "rec_dict.txt"):
        (fixture / name).write_bytes(b"fixture")

    artifact = tmp_path / "2026.07.31-110827"
    artifact.mkdir()
    for name in ("rapidocr.yaml", "det.onnx", "rec.onnx", "rec_dict.txt"):
        (artifact / name).write_bytes(b"artifact")
    (artifact / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "model": "pp-ocrv6-small", "version": "2026.07.31-110827"}),
        encoding="utf-8",
    )

    assert discover_candidate_artifact(tmp_path) == (artifact, "2026.07.31-110827")


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
