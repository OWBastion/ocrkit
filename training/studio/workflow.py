"""Local Studio workers: bounded JEV suggestions and the existing Kaggle runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import subprocess
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

from training.jev.records import OPTION_NONE_CORRECT, OPTION_NOT_VALID, compute_input_digest, record_from_review_row
from training.jev.runner import OmniMlxRunner
from training.studio.core import (
    DEFAULT_WORK_ROOT,
    adopt_engine_agreement,
    adopt_jev_suggestions,
    engines_agree,
    create_batch,
    generate_candidates,
    imported_object_keys,
    load_manifest,
    review_counts,
    review_rows,
    save_jev_suggestion,
)
from training.studio.feedback import get_feedback
from training.studio.r2 import StudioR2Store

ROOT = Path(__file__).resolve().parents[2]


def jev_config() -> dict[str, object]:
    model = Path(os.environ.get("OCRKIT_JEV_MODEL_DIR", str(ROOT / "training/.work/jev/Jev-Omni-MLX-4bit"))).resolve()
    python = Path(os.environ.get("OCRKIT_JEV_WORKER_PYTHON", str(ROOT / "training/.work/jev/venv/bin/python"))).absolute()
    return {"configured": (model / "config.json").is_file() and python.is_file(), "model_dir": str(model), "worker_python": str(python)}


def _write_result(run_dir: Path, result: dict[str, object]) -> None:
    temporary = run_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "result.json")


def run_jev(batch_dir: Path, run_dir: Path, image_tokens: int, on_progress: Callable[[int, int, int], None] | None = None) -> int:
    config = jev_config()
    if not config["configured"]:
        raise ValueError("local Jev-Omni model or worker Python is not configured")
    runner = OmniMlxRunner(Path(str(config["model_dir"])), Path(str(config["worker_python"])))
    processed = saved = errors = 0
    try:
        with (run_dir / "decisions.jsonl").open("a", encoding="utf-8") as decisions:
            for split in ("train", "holdout"):
                for row in review_rows(batch_dir, split, "pending"):
                    if engines_agree(row):
                        continue
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
                    if on_progress:
                        on_progress(processed, saved, errors)
    finally:
        runner.close()
    return 1 if errors else 0


def run_prepare(work_root: Path, run_dir: Path, days: int, holdout_ratio: float, image_tokens: int) -> int:
    """New R2 screenshots → batch → crops/candidates → JEV → confident Train labels adopted."""
    store = StudioR2Store.from_settings()
    if store is None:
        raise ValueError("Studio R2 未配置")
    state: dict[str, object] = {}
    lock = threading.Lock()

    def report(**fields: object) -> None:
        with lock:
            state.update(fields)
            _write_result(run_dir, state)

    report(stage="listing")
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    known = imported_object_keys(work_root)
    fresh = [item for item in store.list_recent(since) if item["key"] not in known][: store.max_objects]
    report(new_screenshots=len(fresh))
    if not fresh:
        batches = sorted(path.parent.name for path in (work_root / "batches").glob("*/batch.json"))
        report(stage="done", batch_id=batches[-1] if batches else None)
        return 0

    marks = get_feedback(since)["marks"]
    keys = [str(item["key"]) for item in fresh]
    temporary = Path(tempfile.mkdtemp(prefix="studio-prepare-", dir=work_root))
    try:
        report(stage="downloading", completed=0, total=len(keys))
        downloaded = store.download_images(keys, temporary, progress_callback=lambda done, total, _key: report(stage="downloading", completed=done, total=total))
        provenance: dict[str, dict[str, object]] = {}
        for item in downloaded:
            entry = dict(item.provenance)
            if entry["object_key"] in marks:
                entry["accuracy"] = marks[entry["object_key"]]
            provenance[str(entry["sha256"])] = entry
        report(stage="candidates", completed=0, total=0)
        batch_dir, _ = create_batch([item.path for item in downloaded], work_root=work_root, holdout_ratio=holdout_ratio, provenance_by_digest=provenance)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    report(batch_id=batch_dir.name)
    generate_candidates(batch_dir)

    report(agreement_adopted=adopt_engine_agreement(batch_dir))
    pending = sum(not engines_agree(row) for split in ("train", "holdout") for row in review_rows(batch_dir, split, "pending"))
    if pending and jev_config()["configured"]:
        report(stage="jev", completed=0, total=pending, jev_errors=0)
        jev_run = run_dir / "jev"
        jev_run.mkdir(exist_ok=True)
        run_jev(batch_dir, jev_run, image_tokens, lambda processed, _saved, errors: report(stage="jev", completed=processed, total=pending, jev_errors=errors))
        report(jev_adopted=adopt_jev_suggestions(batch_dir))
    else:
        report(jev_adopted=None)
    report(stage="done", review=review_counts(batch_dir))
    return 0


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
    parser.add_argument("kind", choices=("prepare", "jev", "kaggle"))
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-tokens", type=int, default=70, choices=(10, 20, 35, 70, 140, 280))
    args = parser.parse_args()
    exit_code = 1
    try:
        if args.kind == "prepare":
            exit_code = run_prepare(args.work_root, args.run_dir, args.days, args.holdout_ratio, args.image_tokens)
        elif args.kind == "jev":
            exit_code = run_jev(args.batch_dir, args.run_dir, args.image_tokens)
            result_path = args.run_dir / "result.json"
            previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            _write_result(args.run_dir, {**previous, "jev_adopted": adopt_jev_suggestions(args.batch_dir)})
        else:
            exit_code = run_kaggle(args.batch_dir, args.run_dir, args.epochs)
        return exit_code
    finally:
        result_path = args.run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        result.update({"workflow_status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code})
        _write_result(args.run_dir, result)


if __name__ == "__main__":
    raise SystemExit(main())
