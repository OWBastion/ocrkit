from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_context
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.service import extract_structured


DEFAULT_RUN_CODE_CASES = Path("tests/fixtures/run_code/cases.json")
DEFAULT_RUN_CODE_IMAGES_DIR = Path("tests/fixtures/run_code")


def evaluate(cases_path: Path, images_dir: Path, model_config: Path | None = None) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    context = create_context()
    if model_config is not None:
        context.ocr_engine = RapidOcrEngine(config_path=model_config)
    total_fields = 0
    matched_fields = 0
    field_counts: dict[str, dict[str, int]] = {}
    field_metrics: dict[str, dict[str, int]] = {}
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
            request_id=f"fixture:{case['id']}",
            engine_name=context.engine_name,
            model_version=context.model_version,
            layout_version=context.layout_version,
        )
        elapsed = (time.perf_counter() - started) * 1000
        actual = response.data.model_dump() if response.data else {}
        expected = case["expected"]
        matched = sum(actual.get(name) == value for name, value in expected.items())
        for name, value in expected.items():
            counts = field_counts.setdefault(name, {"matched": 0, "total": 0})
            counts["total"] += 1
            if actual.get(name) == value:
                counts["matched"] += 1
            metric = field_metrics.setdefault(name, {"matched": 0, "total": 0})
            metric["total"] += 1
            metric["matched"] += actual.get(name) == value
        total_fields += len(expected)
        matched_fields += matched
        elapsed_ms.append(elapsed)
        results.append(
            {
                "id": case["id"],
                "matched_fields": matched,
                "total_fields": len(expected),
                "fields": {
                    name: {
                        "expected": expected_value,
                        "actual": actual.get(name),
                        "matched": actual.get(name) == expected_value,
                    }
                    for name, expected_value in expected.items()
                },
                "layout_version": getattr(response, "layout_version", context.layout_version),
                "quality_warnings": list(getattr(getattr(response, "quality", None), "warnings", [])),
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
        "field_counts": field_counts,
        "field_metrics": {
            name: {**metric, "accuracy": metric["matched"] / metric["total"] if metric["total"] else 0.0}
            for name, metric in sorted(field_metrics.items())
        },
        "mean_elapsed_ms": round(sum(elapsed_ms) / len(elapsed_ms), 2) if elapsed_ms else 0.0,
        "p95_elapsed_ms": round(ordered[p95_index], 2) if ordered else 0.0,
        "results": results,
    }


def main() -> None:
    parser = ArgumentParser(description="Evaluate OCRKit against the checked-in challenge fixtures.")
    parser.add_argument("--cases", type=Path, default=Path("datasets/fixtures/challenge/cases.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/fixtures/challenge"))
    parser.add_argument("--run-code-cases", type=Path, default=DEFAULT_RUN_CODE_CASES)
    parser.add_argument("--run-code-images-dir", type=Path, default=DEFAULT_RUN_CODE_IMAGES_DIR)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-field-accuracy", type=float)
    parser.add_argument("--min-run-code-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--only-run-code",
        action="store_true",
        help="evaluate only the public run-code fixtures once, without the private challenge corpus",
    )
    args = parser.parse_args()
    if args.only_run_code:
        result = evaluate(args.run_code_cases, args.run_code_images_dir, args.model_config)
        run_code_result = result
    else:
        result = evaluate(args.cases, args.images_dir, args.model_config)
        run_code_result = evaluate(args.run_code_cases, args.run_code_images_dir, args.model_config)
        result["run_code"] = run_code_result
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.only_run_code and args.min_field_accuracy is not None and result["field_accuracy"] < args.min_field_accuracy:
        raise SystemExit(
            f"fixture field accuracy {result['field_accuracy']:.6f} is below {args.min_field_accuracy:.6f}"
        )
    if args.min_run_code_accuracy is not None and run_code_result["field_accuracy"] < args.min_run_code_accuracy:
        raise SystemExit(
            "run-code fixture exact-match accuracy "
            f"{run_code_result['field_accuracy']:.6f} is below {args.min_run_code_accuracy:.6f}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
