from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterable
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
) -> dict[str, int]:
    """Create editable recognition-label candidates without changing fixture files."""
    cases = load_cases(cases_path)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}; remove it after review or choose another path")

    ocr = ocr_factory()
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
            split = split_for_case(case_id)

            for roi_name, roi in rois.items():
                processed = preprocess_by_roi(roi_name, roi)
                result = ocr(processed, use_cls=False)
                boxes = result.boxes if result.boxes is not None else []
                texts = result.txts if result.txts is not None else []
                scores = result.scores if result.scores is not None else []
                for index, (box, text, score) in enumerate(zip(boxes, texts, scores, strict=True)):
                    crop = _crop_line(processed, box)
                    candidate = str(text).strip()
                    if crop is None or not candidate:
                        continue
                    if crop.ndim == 2:
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    relative_crop = Path("images") / split / case_id / roi_name / f"{index:03d}.png"
                    crop_path = output_dir / relative_crop
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(crop_path), crop):
                        raise RuntimeError(f"failed to write crop: {crop_path}")
                    rows[split].append(
                        {
                            "crop": relative_crop.as_posix(),
                            "source_id": case_id,
                            "split": split,
                            "roi": roi_name,
                            "box": np.asarray(box, dtype=float).round(2).tolist(),
                            "candidate_text": candidate,
                            "confidence": round(float(score), 4),
                            "review_status": "pending",
                            "transcription": None,
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
        "train_cases": sum(split_for_case(str(case["id"])) == "train" for case in cases),
        "holdout_cases": sum(split_for_case(str(case["id"])) == "holdout" for case in cases),
        "train_candidates": len(rows["train"]),
        "holdout_candidates": len(rows["holdout"]),
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
