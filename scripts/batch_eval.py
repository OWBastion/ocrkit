from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2

from app.main import create_context
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.service import extract_structured


def evaluate(cases_path: Path, images_dir: Path, model_config: Path | None = None) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    context = create_context()
    if model_config is not None:
        context.ocr_engine = RapidOcrEngine(config_path=model_config)
    total_fields = 0
    matched_fields = 0
    elapsed_ms: list[float] = []
    results: list[dict[str, object]] = []

    for case in cases:
        image_path = images_dir / case["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read fixture image: {image_path}")
        started = time.perf_counter()
        response = extract_structured(
            image,
            context.roi_config,
            context.map_names,
            context.map_aliases,
            context.ocr_engine,
            include_debug=False,
        )
        elapsed = (time.perf_counter() - started) * 1000
        actual = response.data.model_dump() if response.data else {}
        expected = case["expected"]
        matched = sum(actual.get(name) == value for name, value in expected.items())
        total_fields += len(expected)
        matched_fields += matched
        elapsed_ms.append(elapsed)
        results.append(
            {
                "id": case["id"],
                "matched_fields": matched,
                "total_fields": len(expected),
                "elapsed_ms": round(elapsed, 2),
            }
        )

    ordered = sorted(elapsed_ms)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return {
        "cases": len(cases),
        "field_accuracy": matched_fields / total_fields if total_fields else 0.0,
        "matched_fields": matched_fields,
        "total_fields": total_fields,
        "mean_elapsed_ms": round(sum(elapsed_ms) / len(elapsed_ms), 2) if elapsed_ms else 0.0,
        "p95_elapsed_ms": round(ordered[p95_index], 2) if ordered else 0.0,
        "results": results,
    }


def main() -> None:
    parser = ArgumentParser(description="Evaluate OCRKit against the checked-in challenge fixtures.")
    parser.add_argument("--cases", type=Path, default=Path("datasets/fixtures/challenge/cases.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/fixtures/challenge"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--min-field-accuracy", type=float)
    args = parser.parse_args()
    result = evaluate(args.cases, args.images_dir, args.model_config)
    if args.min_field_accuracy is not None and result["field_accuracy"] < args.min_field_accuracy:
        raise SystemExit(
            f"fixture field accuracy {result['field_accuracy']:.6f} is below {args.min_field_accuracy:.6f}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
