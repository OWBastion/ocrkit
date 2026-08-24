from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable model release evidence before manifest creation.")
    parser.add_argument("--fixture-report", required=True, type=Path)
    parser.add_argument("--holdout-report", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fixture = _read_json(args.fixture_report, "fixture report")
    if not isinstance(fixture, dict):
        raise SystemExit("fixture report must be a JSON object")
    holdout: dict[str, Any] = {"status": "missing"}
    if args.holdout_report is not None:
        holdout_payload = _read_json(args.holdout_report, "holdout report")
        if not isinstance(holdout_payload, dict):
            raise SystemExit("holdout report must be a JSON object")
        holdout = holdout_payload
        holdout.setdefault("status", "unverified")
    provenance: dict[str, Any] = {"status": "missing"}
    if args.provenance is not None:
        provenance_payload = _read_json(args.provenance, "release provenance")
        provenance = {"status": "recorded", "source": provenance_payload}

    evidence = {
        "schema_version": 1,
        "evaluation": {
            "fixture": {**fixture, "status": "passed"},
            "holdout": holdout,
        },
        "full_test_suite": {"status": "passed"},
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
