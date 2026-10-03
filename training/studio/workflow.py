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
    adopt_jev_suggestions,
    engines_agree,
    create_batch,
    generate_candidates,
    imported_object_keys,
    load_manifest,
    merge_batch_datasets,
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
    report(agreement_adopted=generate_candidates(batch_dir).get("agreement_adopted", 0))
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


def run_kaggle(batch_dir: Path, run_dir: Path, epochs: int, base_checkpoint: Path | None = None, include_batches: tuple[str, ...] = (), base_name: str | None = None) -> int:
    """Train on Kaggle from the chosen base weights over this batch plus any other finalized batches."""
    output = run_dir / "output"
    labels_dir = batch_dir / "dataset"
    data: dict[str, object] = {"batches": [batch_dir.name, *include_batches]}
    if include_batches:
        merged = merge_batch_datasets([batch_dir, *(batch_dir.parent / batch for batch in include_batches)], run_dir / "dataset")
        labels_dir = run_dir / "dataset"
        data.update(train_samples=merged["train_samples"], holdout_samples=merged["holdout_samples"], per_batch=merged["batches"])
    else:
        lines = lambda path: len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0
        data.update(train_samples=lines(labels_dir / "labels/train.txt"), holdout_samples=lines(labels_dir / "labels/holdout.txt"))
    info: dict[str, object] = {
        "base": {"kind": "official"} if base_checkpoint is None else {"kind": "checkpoint", "name": base_name or base_checkpoint.name, "path": str(base_checkpoint)},
        "data": data,
        "dataset_dir": str(labels_dir),
    }
    _write_result(run_dir, info)
    command = [sys.executable, "-u", str(ROOT / "training/run_rec_kaggle.py"), "--labels-dir", str(labels_dir), "--epochs", str(epochs), "--output-dir", str(output)]
    if base_checkpoint is not None:
        command.extend(["--pretrained-checkpoint", str(base_checkpoint)])
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    finally:
        result: dict[str, object] = {**info, "output_run_dir": str(output)}
        status_path = output / "status.json"
        if status_path.is_file():
            result.update(json.loads(status_path.read_text(encoding="utf-8")))
        checkpoint = output / "checkpoint/best_accuracy"
        if checkpoint.with_suffix(".pdparams").is_file():
            result["checkpoint"] = str(checkpoint)
        _write_result(run_dir, result)


def find_trained_checkpoint(runs_dir: Path) -> Path | None:
    """The newest Kaggle run that has a trained checkpoint, whether or not its evaluation passed."""
    for run in sorted(runs_dir.glob("kaggle-*"), reverse=True):
        for base in (run / "output", run / "output/partial"):
            if (base / "checkpoint/best_accuracy.pdparams").is_file():
                return base
    return None


def run_evaluate(batch_dir: Path, run_dir: Path) -> int:
    """Score an already trained checkpoint against the release gate without training again.

    A run whose training succeeded but whose evaluation failed keeps its checkpoint under ``partial``; when this
    evaluation passes, that run is completed the same way a passing Kaggle run would have been.
    """
    base = find_trained_checkpoint(batch_dir / "runs")
    if base is None:
        raise ValueError("no trained checkpoint found for this batch")
    output = base.parent if base.name == "partial" else base
    evaluation = output / f"evaluation-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    print(f"Evaluating {base / 'checkpoint/best_accuracy'} (no training)", flush=True)
    dataset = batch_dir / "dataset"
    training_result = output.parent / "result.json"
    if training_result.is_file():
        dataset = Path(json.loads(training_result.read_text(encoding="utf-8")).get("dataset_dir") or dataset)
    command = [str(ROOT / "training/evaluate_rec_checkpoint.sh"), str((base / "checkpoint/best_accuracy").resolve()), str(evaluation.resolve())]
    if (dataset / "labels/holdout.txt").is_file():
        command.extend(["--holdout-labels", str((dataset / "labels/holdout.txt").resolve()), "--holdout-images-root", str(dataset.resolve())])
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    report_path = evaluation / "fixture_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    inherited = {key: value for key, value in (json.loads(training_result.read_text(encoding="utf-8")) if training_result.is_file() else {}).items() if key in {"base", "data", "dataset_dir"}}
    _write_result(run_dir, {**inherited, "evaluated_checkpoint": str(base / "checkpoint/best_accuracy"), "field_accuracy": report.get("field_accuracy"), "evaluation_dir": str(evaluation)})
    if status:
        return status

    stamp = datetime.now(UTC).strftime("%H%M%S")
    if base != output:
        if (base / "evaluation").exists():
            (base / "evaluation").rename(base / "evaluation-failed")
        for child in list(base.iterdir()):
            target = output / child.name
            if target.exists():
                target.rename(output / f"{child.name}-previous-{stamp}")
            child.rename(target)
        base.rmdir()
    elif (output / "evaluation").exists():
        (output / "evaluation").rename(output / f"evaluation-previous-{stamp}")
    final = evaluation.rename(output / "evaluation")
    metadata_path = output / "run.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["evaluation"] = {
            "location": "local",
            "report": f"{final.name}/fixture_report.json",
            "field_accuracy": report.get("field_accuracy"),
            "run_code_accuracy": report.get("run_code", {}).get("field_accuracy"),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status_path = output / "status.json"
    kernel = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    status_path.write_text(json.dumps({**kernel, "status": "success", "error": None}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_result(run_dir, {
        **inherited,
        "evaluated_checkpoint": str(output / "checkpoint/best_accuracy"),
        "checkpoint": str(output / "checkpoint/best_accuracy"),
        "output_run_dir": str(output),
        "field_accuracy": report.get("field_accuracy"),
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("prepare", "jev", "kaggle", "evaluate"))
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--base-checkpoint", type=Path, help="Local .pdparams to train from instead of the official weights")
    parser.add_argument("--base-name", help="Display name recorded for --base-checkpoint")
    parser.add_argument("--include-batch", action="append", default=[], help="Another finalized batch whose labels join the training data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-tokens", type=int, default=70, choices=(10, 20, 35, 70, 140, 280))
    args = parser.parse_args()
    exit_code = 1
    try:
        if args.kind == "prepare":
            exit_code = run_prepare(args.work_root, args.run_dir, args.days, args.holdout_ratio, args.image_tokens)
        elif args.kind == "evaluate":
            exit_code = run_evaluate(args.batch_dir, args.run_dir)
        elif args.kind == "jev":
            exit_code = run_jev(args.batch_dir, args.run_dir, args.image_tokens)
            result_path = args.run_dir / "result.json"
            previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            _write_result(args.run_dir, {**previous, "jev_adopted": adopt_jev_suggestions(args.batch_dir)})
        else:
            exit_code = run_kaggle(args.batch_dir, args.run_dir, args.epochs, args.base_checkpoint, tuple(args.include_batch), args.base_name)
        return exit_code
    finally:
        result_path = args.run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        result.update({"workflow_status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code})
        _write_result(args.run_dir, result)


if __name__ == "__main__":
    raise SystemExit(main())
