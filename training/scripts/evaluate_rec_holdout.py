from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ocr.rapidocr_engine import RapidOcrEngine

MIN_HOLDOUT_ACCURACY = 0.9604221635883905


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def evaluate(labels_path: Path, images_root: Path, model_config: Path, min_accuracy: float) -> dict[str, object]:
    engine = RapidOcrEngine(model_config)
    results: list[dict[str, object]] = []
    matched = 0
    total = 0
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
        actual = engine.recognize(image)
        expected_text = canonicalize(expected)
        actual_text = canonicalize(actual.text)
        is_match = actual_text == expected_text
        matched += int(is_match)
        total += 1
        results.append({
            "crop": relative,
            "matched": is_match,
            "confidence": actual.confidence,
        })
    accuracy = matched / total if total else 0.0
    return {
        "schema_version": 1,
        "status": "passed" if total > 0 and accuracy >= min_accuracy else "failed",
        "accuracy": accuracy,
        "matched": matched,
        "total": total,
        "min_accuracy": min_accuracy,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate RapidOCR artifact against isolated holdout crops.")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-accuracy", type=float, default=MIN_HOLDOUT_ACCURACY)
    args = parser.parse_args()
    report = evaluate(args.labels, args.images_root, args.model_config, args.min_accuracy)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(f"holdout accuracy {report['accuracy']:.6f} is below {args.min_accuracy:.6f}")


if __name__ == "__main__":
    main()
