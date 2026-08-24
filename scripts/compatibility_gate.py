from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.batch_eval import evaluate


DEFAULT_MATRIX = Path("configs/bastion_screenshot_compatibility.json")


def load_matrix(path: Path) -> dict[str, object]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or matrix.get("schema_version") != 1:
        raise ValueError("compatibility matrix must use schema_version 1")
    producer = matrix.get("producer_contract")
    if not isinstance(producer, dict) or not producer.get("revision") or not producer.get("minimum_released_version"):
        raise ValueError("compatibility matrix must identify the Bastion producer contract")
    layouts = matrix.get("supported_layouts")
    if not isinstance(layouts, list) or not layouts:
        raise ValueError("compatibility matrix must declare supported layouts")
    layout_versions = {item.get("layout_version") for item in layouts if isinstance(item, dict)}
    if None in layout_versions:
        raise ValueError("every supported layout must have a layout_version")
    fields = matrix.get("critical_fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("compatibility matrix must declare critical fields")
    fixture_sets = matrix.get("fixture_sets")
    if not isinstance(fixture_sets, list) or not fixture_sets:
        raise ValueError("compatibility matrix must declare fixture sets")
    revisions = {
        str(item["revision"])
        for item in matrix.get("supported_producer_revisions", [])
        if isinstance(item, dict) and item.get("revision")
    }
    if str(producer["revision"]) not in revisions:
        raise ValueError("current producer contract must be in supported producer revisions")
    for fixture_set in fixture_sets:
        if not isinstance(fixture_set, dict) or str(fixture_set.get("producer_revision", producer["revision"])) not in revisions:
            raise ValueError("fixture set references an unsupported producer revision")
    return matrix


def _classify_failure(case: dict[str, object], field: str) -> str:
    warnings = set(case.get("quality_warnings", []))
    if any(warning.startswith("quality.") for warning in warnings):
        return "quality/rejection behavior regression"
    if field == "run_code":
        case_id = str(case.get("id", ""))
        if case_id in {"ambiguous", "malformed"}:
            return "parser/normalization regression"
        if case_id in {"missing", "cropped"}:
            return "quality/rejection behavior regression"
    return "recognition/model accuracy regression"


def run_gate(matrix_path: Path, model_config: Path | None = None) -> dict[str, object]:
    matrix = load_matrix(matrix_path)
    supported_layouts = {
        str(item["layout_version"])
        for item in matrix["supported_layouts"]
        if isinstance(item, dict)
    }
    critical_fields = {str(field) for field in matrix["critical_fields"]}
    reports: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    seen_layouts: set[str] = set()
    for fixture_set in matrix["fixture_sets"]:
        if not isinstance(fixture_set, dict):
            raise ValueError("fixture set must be an object")
        cases_path = Path(str(fixture_set["cases"]))
        images_dir = Path(str(fixture_set["images"]))
        if not cases_path.is_file():
            if fixture_set.get("required", True):
                raise FileNotFoundError(f"required compatibility fixture set is missing: {cases_path}")
            continue
        result = evaluate(cases_path, images_dir, model_config)
        selected_fields = {str(field) for field in fixture_set.get("critical_fields", critical_fields)}
        field_counts = result.get("field_counts", {})
        baseline_only = bool(fixture_set.get("baseline_only", False))
        for field in sorted(selected_fields):
            counts = field_counts.get(field, {"matched": 0, "total": 0})
            if counts["total"] == 0:
                if not baseline_only:
                    failures.append({"fixture_set": str(fixture_set["id"]), "field": field, "classification": "parser/normalization regression"})
                continue
            if counts["matched"] != counts["total"]:
                for case in result["results"]:
                    field_result = case.get("fields", {}).get(field)
                    if isinstance(field_result, dict) and not field_result.get("matched") and not baseline_only:
                        failures.append(
                            {
                                "fixture_set": str(fixture_set["id"]),
                                "case": str(case["id"]),
                                "field": field,
                                "classification": _classify_failure(case, field),
                            }
                        )
        unknown_layouts = sorted({str(case["layout_version"]) for case in result["results"]} - supported_layouts)
        seen_layouts.update(str(case["layout_version"]) for case in result["results"])
        for layout_version in unknown_layouts:
            failures.append(
                {
                    "fixture_set": str(fixture_set["id"]),
                    "layout_version": layout_version,
                    "classification": "unsupported/wrong layout selection",
                }
            )
        required_layouts = {str(layout) for layout in fixture_set.get("required_layouts", supported_layouts)}
        missing_layouts = sorted(required_layouts - {str(case["layout_version"]) for case in result["results"]})
        if missing_layouts and not baseline_only:
            failures.extend(
                {
                    "fixture_set": str(fixture_set["id"]),
                    "layout_version": layout_version,
                    "classification": "unsupported/wrong layout selection",
                }
                for layout_version in missing_layouts
            )
        reports.append({
            "id": fixture_set["id"],
            "producer_revision": fixture_set.get("producer_revision", matrix["producer_contract"]["revision"]),
            "baseline_only": baseline_only,
            "result": result,
            "critical_fields": sorted(selected_fields),
            "required_layouts": sorted(required_layouts),
            "missing_layouts": missing_layouts,
        })
    return {
        "schema_version": 1,
        "producer_contract": matrix["producer_contract"],
        "supported_layouts": matrix["supported_layouts"],
        "critical_fields": sorted(critical_fields),
        "fixture_sets": reports,
        "failures": failures,
        "ok": not failures,
    }


def main() -> None:
    parser = ArgumentParser(description="Gate OCRKit releases against supported Bastion screenshot revisions.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_gate(args.matrix, args.model_config)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("Bastion screenshot compatibility gate failed")


if __name__ == "__main__":
    main()
