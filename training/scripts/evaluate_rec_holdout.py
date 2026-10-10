from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ocr.rapidocr_engine import RapidOcrEngine

MEASURE = "recognition-only exact match, whitespace ignored"


def canonicalize(text: str) -> str:
    """Spacing around digits and CJK is a rendering detail of the HUD that OCR and labelling disagree about."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def evaluate(labels_path: Path, images_root: Path, model_config: Path) -> dict[str, object]:
    """Score only the recognition model on isolated line crops.

    The crops are already single text lines, so the detector must not run: on a tight crop it finds nothing, which
    made every model score the same.
    """
    engine = RapidOcrEngine(model_config)._engine
    results: list[dict[str, object]] = []
    by_roi: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    matched = 0
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative, expected = line.split("\t", 1)
        image_path = (images_root / relative).resolve()
        if images_root.resolve() not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"holdout crop is outside the images root or missing: {relative}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read holdout crop: {image_path}")
        recognized = engine(image, use_det=False, use_cls=False)
        actual = "".join(recognized.txts) if recognized.txts else ""
        is_match = canonicalize(actual) == canonicalize(expected)
        matched += int(is_match)
        parts = relative.split("/")
        roi = parts[-2] if len(parts) >= 2 else "unknown"
        by_roi[roi][0] += int(is_match)
        by_roi[roi][1] += 1
        results.append({"crop": relative, "matched": is_match})
    total = len(results)
    return {
        "schema_version": 2,
        "measure": MEASURE,
        "accuracy": matched / total if total else 0.0,
        "matched": matched,
        "total": total,
        "by_roi": {roi: {"matched": m, "total": t} for roi, (m, t) in sorted(by_roi.items())},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a RapidOCR artifact's recognition model on isolated holdout line crops.")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.labels, args.images_root, args.model_config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"holdout {report['matched']}/{report['total']} = {report['accuracy']:.4f} ({MEASURE})")


if __name__ == "__main__":
    main()
