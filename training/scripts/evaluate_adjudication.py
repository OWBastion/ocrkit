from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser.terminology import default_terminology_catalog
from training.adjudication.adjudicator import build_adjudicator
from training.adjudication.evaluate import (
    DEFAULT_GATE,
    load_replay_outputs,
    run_experiment,
    write_report,
)
from training.adjudication.records import load_records


def _load_gate(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_GATE)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic terminology normalization alone against "
        "normalization plus a constrained text adjudicator, using reviewed annotations."
    )
    parser.add_argument("--records", type=Path, required=True, help="JSONL of ReviewedAnnotationRecord (from #5 import or fixtures)")
    parser.add_argument("--report", type=Path, required=True, help="Path for the JSON report; captured outputs go next to it")
    parser.add_argument("--adjudicator", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--replay-dir", type=Path, help="Directory of captured outputs to replay instead of calling the adjudicator")
    parser.add_argument("--gate-json", type=Path, help="JSON override for the go/no-go gate")
    parser.add_argument("--confidence-gate", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--without-terminology-catalog", action="store_true", help="Do not recompute #3 normalization from configs")
    args = parser.parse_args()

    records = load_records(args.records)
    if not records:
        raise SystemExit(f"no records found in {args.records}")

    adjudicator = build_adjudicator(
        args.adjudicator,
        endpoint=os.environ.get("OCRKIT_ADJUDICATION_ENDPOINT"),
        model=os.environ.get("OCRKIT_ADJUDICATION_MODEL"),
        api_key=os.environ.get("OCRKIT_ADJUDICATION_API_KEY"),
        prompt_version=args.prompt_version,
        timeout_seconds=args.timeout,
    )
    catalog = None if args.without_terminology_catalog else default_terminology_catalog()
    replay = load_replay_outputs(args.replay_dir) if args.replay_dir else None
    provider = adjudicator if args.adjudicator == "openai" else None

    result = run_experiment(
        records,
        catalog,
        adjudicator,
        gate=_load_gate(args.gate_json),
        confidence_gate=args.confidence_gate,
        replay_outputs=replay,
        provider=provider,
    )

    report_path = args.report
    write_report(report_path, result, result.captured_outputs)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
