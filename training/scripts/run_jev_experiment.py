from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.jev.evaluate import (
    DEFAULT_GATE,
    load_decisions,
    run_experiment,
    write_decisions,
    write_report,
)
from training.jev.records import load_records, load_studio_records, write_records
from training.jev.runner import build_runner

DEFAULT_BATCHES_ROOT = ROOT / "training/.work/studio/batches"
DEFAULT_WORKER_PYTHON = ROOT / "training/.work/jev/venv/bin/python"


def _load_gate(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_GATE)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Studio-reviewed ROI rows through the Jev-Omni bounded "
        "pre-review task and measure review reduction, auto-decision accuracy, "
        "false-confident errors, and resource cost (#25). Read-only against "
        "batches; Jev output never enters labels, training, or recognition."
    )
    parser.add_argument("--batches-root", type=Path, default=DEFAULT_BATCHES_ROOT)
    parser.add_argument("--records", type=Path, help="Load a saved records.jsonl instead of scanning batches")
    parser.add_argument("--records-out", type=Path, help="Write materialized records JSONL for replay")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, help="JSONL of captured decisions; merged with new outputs")
    parser.add_argument("--replay", type=Path, help="Reuse a decisions JSONL instead of calling the runner")
    parser.add_argument("--runner", choices=("mock", "omni-mlx"), required=True)
    parser.add_argument("--model-dir", type=Path, help="Local Jev-Omni-MLX-4bit checkout")
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--calibration", type=Path, help="Optional temperature-scaling JSON for the worker")
    parser.add_argument("--image-tokens", type=int, nargs="+", default=[70], choices=[10, 20, 35, 70, 140, 280])
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--max-rows", type=int, help="Deterministic stride subsample of residual rows")
    args = parser.parse_args()

    if args.records:
        records = load_records(args.records)
        crop_paths = {}
        if args.runner == "omni-mlx":
            live_records, live_paths = load_studio_records(args.batches_root)
            wanted = {record.record_id for record in records}
            crop_paths = {key: value for key, value in live_paths.items() if key in wanted}
            missing = wanted - set(crop_paths)
            if missing:
                raise SystemExit(f"{len(missing)} saved records have no live crop under {args.batches_root}")
    else:
        records, crop_paths = load_studio_records(args.batches_root)
    if not records:
        raise SystemExit(f"no review rows found under {args.batches_root}")
    if args.records_out:
        write_records(args.records_out, records)

    runner = build_runner(
        args.runner,
        model_dir=args.model_dir,
        worker_python=args.worker_python,
        calibration=args.calibration,
        timeout_seconds=args.timeout,
    )
    decisions_path = args.decisions or args.report_dir / "decisions.jsonl"
    replay = load_decisions(args.replay) if args.replay else load_decisions(decisions_path)
    if replay:
        print(f"resuming: {len(replay)} captured decisions from {decisions_path}", flush=True)

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_log = decisions_path.open("a", encoding="utf-8")
    total_calls = [0]

    def progress(record, output):
        total_calls[0] += 1
        decisions_log.write(
            json.dumps(
                {"input_digest": record.input_digest, "image_tokens": output.image_tokens, "output": output.model_dump()},
                ensure_ascii=False,
            )
            + "\n"
        )
        decisions_log.flush()
        if total_calls[0] % 25 == 0 or output.status != "ok":
            print(
                f"[{total_calls[0]}] {record.record_id} tokens={output.image_tokens} "
                f"-> {output.selected_option!r} conf={output.confidence} status={output.status}",
                flush=True,
            )

    try:
        result = run_experiment(
            records,
            crop_paths,
            runner,
            image_tokens_list=args.image_tokens,
            gate=_load_gate(args.gate_json),
            fit_fraction=args.fit_fraction,
            replay_outputs=replay,
            max_rows=args.max_rows,
            progress=progress,
        )
    finally:
        decisions_log.close()

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    result.metrics["code_revision"] = revision or None
    write_report(args.report_dir, result.metrics)
    merged = dict(replay or {})
    merged.update({key: output.model_dump() for key, output in result.outputs.items()})
    write_decisions(decisions_path, merged)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    print(f"report written to {args.report_dir / 'report.json'}")
    print(f"decisions written to {decisions_path}")


if __name__ == "__main__":
    main()
