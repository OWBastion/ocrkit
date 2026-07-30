from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.roi_config import RoiConfig, load_roi_config
from app.image.loader import decode_image
from app.image.preprocess import preprocess_by_roi
from app.image.roi import crop_all_rois
from training.vision import VisionLine, VisionOcr


DEFAULT_FIXTURES = ROOT / "datasets/fixtures/challenge"
DEFAULT_OUTPUT = ROOT / "datasets/labeled/rec"
DEFAULT_ROI_CONFIG = ROOT / "configs/roi_1280x720.yaml"

HOLDOUT_IDS = frozenset(
    {
        "samoa_hell_01",
        "route_66_01",
        "numbani_01",
        "new_junk_city_in_progress_01",
        "circuit_royal_01",
        "hanamura_02",
        "anubis_temple_01",
        "lijiang_tower_01",
    }
)
AUTO_ACCEPT_CONFIDENCE = 0.98
MATCH_IOU = 0.5


@dataclass(frozen=True)
class CandidateLine:
    text: str
    confidence: float
    box: np.ndarray


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases.json must contain a list")
    return data


def split_for_case(case_id: str) -> str:
    return "holdout" if case_id in HOLDOUT_IDS else "train"


def _crop_line(image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        return None
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(image.shape[1], int(np.ceil(points[:, 0].max())))
    y2 = min(image.shape[0], int(np.ceil(points[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def _box_bounds(box: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _box_bounds(first)
    bx1, by1, bx2, by2 = _box_bounds(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def _union_box(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = _box_bounds(first)
    ox1, oy1, ox2, oy2 = _box_bounds(second)
    return np.array(
        [[min(x1, ox1), min(y1, oy1)], [max(x2, ox2), min(y1, oy1)], [max(x2, ox2), max(y2, oy2)], [min(x1, ox1), max(y2, oy2)]],
        dtype=np.float32,
    )


def _rapid_lines(result: Any) -> list[CandidateLine]:
    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    scores = result.scores if result.scores is not None else []
    return [
        CandidateLine(str(text).strip(), float(score), np.asarray(box, dtype=np.float32))
        for box, text, score in zip(boxes, texts, scores, strict=True)
        if str(text).strip()
    ]


def _vision_lines(result: list[VisionLine]) -> list[CandidateLine]:
    return [CandidateLine(line.text, line.confidence, line.box) for line in result]


def _paired_lines(rapid: list[CandidateLine], vision: list[CandidateLine]) -> list[tuple[CandidateLine | None, CandidateLine | None]]:
    pairs: list[tuple[CandidateLine | None, CandidateLine | None]] = []
    unmatched = set(range(len(vision)))
    for rapid_line in rapid:
        index, overlap = max(
            ((index, _iou(rapid_line.box, vision[index].box)) for index in unmatched),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        if overlap >= MATCH_IOU:
            pairs.append((rapid_line, vision[index]))
            unmatched.remove(index)
        else:
            pairs.append((rapid_line, None))
    pairs.extend((None, vision[index]) for index in sorted(unmatched))
    return pairs


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_label_scaffold(output_dir: Path) -> None:
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "holdout"):
        (labels_dir / f"{split}.txt").write_text("", encoding="utf-8")
    (labels_dir / "README.md").write_text(
        "Review review/train.jsonl and review/holdout.jsonl first. For each accepted row, "
        "append `<crop>\\t<transcription>` to the corresponding labels/*.txt file. "
        "Do not include pending or rejected candidates.\n",
        encoding="utf-8",
    )


def prepare_candidates(
    cases_path: Path,
    fixture_dir: Path,
    output_dir: Path,
    roi_config: RoiConfig,
    ocr_factory: Callable[[], Any] = RapidOCR,
    vision_factory: Callable[[], Any] = VisionOcr,
) -> dict[str, int]:
    """Create editable recognition-label candidates without changing fixture files."""
    cases = load_cases(cases_path)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}; remove it after review or choose another path")

    ocr = ocr_factory()
    vision = vision_factory()
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    output_dir.mkdir(parents=True)
    try:
        for case in cases:
            case_id = str(case["id"])
            image_path = fixture_dir / str(case["image"])
            if not image_path.is_file():
                raise FileNotFoundError(f"fixture image does not exist: {image_path}")
            image = decode_image(image_path.read_bytes())
            _, rois = crop_all_rois(image, roi_config)
            requested_split = case.get("split")
            if requested_split is not None and requested_split not in {"train", "holdout"}:
                raise ValueError(f"case {case_id} has invalid split: {requested_split}")
            split = str(requested_split) if requested_split else split_for_case(case_id)

            for roi_name, roi in rois.items():
                processed = preprocess_by_roi(roi_name, roi)
                rapid = _rapid_lines(ocr(processed, use_cls=False))
                vision_lines = _vision_lines(vision.recognize(processed))
                for index, (rapid_line, vision_line) in enumerate(_paired_lines(rapid, vision_lines)):
                    if rapid_line is not None and vision_line is not None:
                        box = _union_box(rapid_line.box, vision_line.box)
                    else:
                        box = (rapid_line or vision_line).box
                    crop = _crop_line(processed, box)
                    if crop is None:
                        continue
                    if crop.ndim == 2:
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    relative_crop = Path("images") / split / case_id / roi_name / f"{index:03d}.png"
                    crop_path = output_dir / relative_crop
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(crop_path), crop):
                        raise RuntimeError(f"failed to write crop: {crop_path}")
                    rapid_text = rapid_line.text if rapid_line else None
                    vision_text = vision_line.text if vision_line else None
                    auto_accepted = (
                        rapid_line is not None
                        and vision_line is not None
                        and rapid_line.confidence >= AUTO_ACCEPT_CONFIDENCE
                        and vision_line.confidence >= AUTO_ACCEPT_CONFIDENCE
                        and canonicalize(rapid_line.text) == canonicalize(vision_line.text)
                    )
                    rows[split].append(
                        {
                            "crop": relative_crop.as_posix(),
                            "source_id": case_id,
                            "split": split,
                            "roi": roi_name,
                            "box": np.asarray(box, dtype=float).round(2).tolist(),
                            "candidate_text": rapid_text or vision_text,
                            "confidence": round(max(line.confidence for line in (rapid_line, vision_line) if line), 4),
                            "rapidocr_text": rapid_text,
                            "rapidocr_confidence": round(rapid_line.confidence, 4) if rapid_line else None,
                            "vision_text": vision_text,
                            "vision_confidence": round(vision_line.confidence, 4) if vision_line else None,
                            "review_status": "accepted" if auto_accepted else "pending",
                            "transcription": canonicalize(rapid_line.text) if auto_accepted else None,
                            "auto_accept_reason": "rapidocr_vision_agreement" if auto_accepted else None,
                        }
                    )

        review_dir = output_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for split, split_rows in rows.items():
            _write_jsonl(review_dir / f"{split}.jsonl", split_rows)
        _write_label_scaffold(output_dir)
    except Exception:
        shutil.rmtree(output_dir)
        raise

    return {
        "cases": len(cases),
        "train_cases": sum((str(case.get("split")) if case.get("split") else split_for_case(str(case["id"]))) == "train" for case in cases),
        "holdout_cases": sum((str(case.get("split")) if case.get("split") else split_for_case(str(case["id"]))) == "holdout" for case in cases),
        "train_candidates": len(rows["train"]),
        "holdout_candidates": len(rows["holdout"]),
        "auto_accepted": sum(row["review_status"] == "accepted" for split_rows in rows.values() for row in split_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RapidOCR-assisted PP-OCR rec review candidates from fixtures.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_candidates(
                args.fixtures / "cases.json",
                args.fixtures,
                args.output,
                load_roi_config(args.roi_config),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
