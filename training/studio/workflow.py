"""Local Studio workers: bounded JEV suggestions and the existing Kaggle runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path

from training.jev.records import OPTION_NONE_CORRECT, OPTION_NOT_VALID, compute_input_digest, record_from_review_row
from training.jev.runner import OmniMlxRunner
from training.studio.core import load_manifest, review_rows, save_jev_suggestion

ROOT = Path(__file__).resolve().parents[2]


def jev_config() -> dict[str, object]:
    model = Path(os.environ.get("OCRKIT_JEV_MODEL_DIR", str(ROOT / "training/.work/jev/Jev-Omni-MLX-4bit"))).resolve()
    python = Path(os.environ.get("OCRKIT_JEV_WORKER_PYTHON", str(ROOT / "training/.work/jev/venv/bin/python"))).absolute()
    return {"configured": (model / "config.json").is_file() and python.is_file(), "model_dir": str(model), "worker_python": str(python)}


def _write_result(run_dir: Path, result: dict[str, object]) -> None:
    temporary = run_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "result.json")


def run_jev(batch_dir: Path, run_dir: Path, image_tokens: int) -> int:
    config = jev_config()
    if not config["configured"]:
        raise ValueError("local Jev-Omni model or worker Python is not configured")
    runner = OmniMlxRunner(Path(str(config["model_dir"])), Path(str(config["worker_python"])))
    processed = saved = errors = 0
    try:
        with (run_dir / "decisions.jsonl").open("a", encoding="utf-8") as decisions:
            for split in ("train", "holdout"):
                for row in review_rows(batch_dir, split, "pending"):
                    crop = str(row["crop"])
                    dataset = (batch_dir / "dataset").resolve()
                    image = (dataset / crop).resolve()
                    if dataset not in image.parents or not image.is_file():
                        raise ValueError("pending review row has a missing or unsafe crop")
                    record = record_from_review_row(batch_dir.name, split, load_manifest(batch_dir).get("layout_version"), hashlib.sha256(image.read_bytes()).hexdigest(), row)
                    record.input_digest = compute_input_digest(record)
                    output = runner.decide(record, image, image_tokens)
                    suggestion = output.model_dump()
                    option = output.selected_option
                    action = "error" if output.status != "ok" else "accept" if option in record.candidate_options else "reject" if option == OPTION_NOT_VALID else "manual"
                    if option == OPTION_NONE_CORRECT:
                        action = "manual"
                    suggestion["action"] = action
                    decisions.write(json.dumps(suggestion, ensure_ascii=False) + "\n")
                    decisions.flush()
                    saved += int(save_jev_suggestion(batch_dir, split, crop, suggestion))
                    processed += 1
                    errors += int(output.status != "ok")
                    result = {"processed": processed, "saved": saved, "errors": errors}
                    _write_result(run_dir, result)
                    print(f"JEV suggestions: {processed} processed, {saved} saved, {errors} errors", flush=True)
    finally:
        runner.close()
    return 1 if errors else 0


def run_kaggle(batch_dir: Path, run_dir: Path, epochs: int) -> int:
    output = run_dir / "output"
    command = [sys.executable, "-u", str(ROOT / "training/run_rec_kaggle.py"), "--labels-dir", str(batch_dir / "dataset"), "--epochs", str(epochs), "--output-dir", str(output)]
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    finally:
        result: dict[str, object] = {"output_run_dir": str(output)}
        status_path = output / "status.json"
        if status_path.is_file():
            result.update(json.loads(status_path.read_text(encoding="utf-8")))
        checkpoint = output / "checkpoint/best_accuracy"
        if checkpoint.with_suffix(".pdparams").is_file():
            result["checkpoint"] = str(checkpoint)
        _write_result(run_dir, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("jev", "kaggle"))
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-tokens", type=int, default=70, choices=(10, 20, 35, 70, 140, 280))
    args = parser.parse_args()
    exit_code = 1
    try:
        exit_code = run_jev(args.batch_dir, args.run_dir, args.image_tokens) if args.kind == "jev" else run_kaggle(args.batch_dir, args.run_dir, args.epochs)
        return exit_code
    finally:
        result_path = args.run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        result.update({"workflow_status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code})
        _write_result(args.run_dir, result)


if __name__ == "__main__":
    raise SystemExit(main())
