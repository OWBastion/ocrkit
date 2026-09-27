#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    # Running this file directly (python training/run_rec_colab.py) puts training/, not the
    # repo root, on sys.path; the shared training.* package needs the root added explicitly.
    sys.path.insert(0, str(_ROOT))

from training.remote_gpu_common import (  # noqa: E402
    ACCEPTED_STAGING,
    OFFICIAL_BASE_CHECKPOINT,
    PADDLE_WHEEL,
    PRETRAINED_CHECKPOINT,
    R2_UPLOAD_URL_BUFFER_SECONDS,
    ROOT,
    capture_command,
    evaluate_locally,
    git_metadata,
    read_dataset_provenance,
    require_r2_store,
    retrieve_checkpoint,
    sha256,
    stage_inputs,
    stream_command,
    transfer_with_retry,
    validate_labels,
)

RUNS = ROOT / "training/.work/colab-runs"
R2_KEY_PREFIX = "colab-runs"
PART_BYTES = 32 * 1024 * 1024
REMOTE_RUNNER = ROOT / "training/colab_remote.py"


def session_still_active(colab: str, session: str, log_path: Path) -> bool:
    """`colab stop` can fail simply because Colab already reclaimed an idle/finished runtime."""
    result = capture_command([colab, "sessions"])
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {colab} sessions\n{result.stdout}{result.stderr}\n")
    if result.returncode:
        return True  # could not verify; assume active so a real leak is not silently dropped
    return session in result.stdout


def upload_archive(colab: str, session: str, archive: Path, run_dir: Path, log_path: Path) -> int:
    """Colab uploads are single JSON requests, so send the archive in bounded parts."""
    parts_dir = run_dir / "input-parts"
    parts_dir.mkdir()
    try:
        with archive.open("rb") as stream:
            for index, chunk in enumerate(iter(lambda: stream.read(PART_BYTES), b"")):
                part = parts_dir / f"part{index:04d}"
                part.write_bytes(chunk)
                status = transfer_with_retry(
                    [colab, "upload", "-s", session, str(part), f"/content/ocrkit-input.part{index:04d}"],
                    log_path,
                    stream_command,
                )
                part.unlink()
                if status:
                    return status
        return 0
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def fetch_remote_metadata(colab: str, session: str, run_dir: Path, log_path: Path) -> tuple[dict[str, Any], Path | None]:
    """Only run.json and remote.log travel through the Colab CLI; both are small, unlike the checkpoint."""
    metadata_path = run_dir / "remote-run.json"
    status = transfer_with_retry(
        [colab, "download", "-s", session, "/content/ocrkit-colab/results/run.json", str(metadata_path)], log_path, stream_command
    )
    if status:
        raise RuntimeError("Colab did not return run metadata (results/run.json)")
    remote_log_path = run_dir / "remote-log.txt"
    if transfer_with_retry(
        [colab, "download", "-s", session, "/content/ocrkit-colab/results/remote.log", str(remote_log_path)], log_path, stream_command
    ):
        remote_log_path = None  # best-effort: diagnostics, not required for the run's outcome
    return json.loads(metadata_path.read_text(encoding="utf-8")), remote_log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the OCRKit recognition model on a Colab GPU, then evaluate it locally.")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "datasets/labeled/rec")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", default="T4", help="Colab GPU preference; no accelerator fallback is attempted")
    parser.add_argument("--timeout-seconds", type=float, default=6 * 3600, help="Upper bound for the remote training run")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    colab = shutil.which("colab")
    if not colab:
        parser.error("Google Colab CLI is required; install it with uv tool install google-colab-cli")

    r2 = require_r2_store(parser)

    dataset_root = args.labels_dir.resolve()
    train_label = dataset_root / "labels/train.txt"
    holdout_label = dataset_root / "labels/holdout.txt"
    for path in (train_label, holdout_label):
        if not path.is_file():
            parser.error(f"required training input is missing: {path}")
    using_official_checkpoint = args.pretrained_checkpoint == PRETRAINED_CHECKPOINT
    if not using_official_checkpoint and not args.pretrained_checkpoint.is_file():
        parser.error(f"required training input is missing: {args.pretrained_checkpoint}")
    train_count, train_images = validate_labels(train_label)
    holdout_count, holdout_images = validate_labels(holdout_label)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid():x}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_archive = run_dir / "ocrkit-input.tar.gz"
    colab_log = run_dir / "colab.log"

    source_revision, source_dirty = git_metadata(ROOT)
    dataset_revision, dataset_dirty = git_metadata(dataset_root)
    try:
        dataset_source = dataset_root.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_source = "external"
    if using_official_checkpoint:
        checkpoint_path = None
        base_checkpoint = {**OFFICIAL_BASE_CHECKPOINT, "source": "official-download"}
    else:
        checkpoint_path = args.pretrained_checkpoint.resolve()
        base_checkpoint = {
            "model": "custom",
            "source": "uploaded",
            "source_name": checkpoint_path.name,
            "sha256": sha256(checkpoint_path),
        }
    checkpoint_upload_key = f"{R2_KEY_PREFIX}/{run_id}/checkpoint.pdparams"
    checkpoint_upload_url = r2.generate_presigned_put_url(
        r2.default_bucket, checkpoint_upload_key, expires_in_seconds=int(args.timeout_seconds) + R2_UPLOAD_URL_BUFFER_SECONDS
    )
    run_request = {
        "run_id": run_id,
        "ocrkit_revision": source_revision,
        "ocrkit_worktree_dirty": source_dirty,
        "dataset": {
            **read_dataset_provenance(dataset_root),
            "source": dataset_source,
            "revision": dataset_revision,
            "worktree_dirty": dataset_dirty,
            "train_samples": train_count,
            "holdout_samples": holdout_count,
        },
        "base_checkpoint": base_checkpoint,
        "paddle_wheel_mirror": PADDLE_WHEEL,
        "checkpoint_upload": {"bucket": r2.default_bucket, "key": checkpoint_upload_key, "url": checkpoint_upload_url},
        "training": {
            "epochs": args.epochs,
            "device": "cuda",
            "gpu_preference": args.gpu,
            "recipe": "training/configs/rec_pp_ocrv6_small.yaml",
            "paddleocr_recipe": "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml",
        },
    }
    try:
        stage_inputs(input_archive, run_request, dataset_root, checkpoint_path, train_images, holdout_images)
    except Exception as exc:
        input_archive.unlink(missing_ok=True)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    session = f"ocrkit-rec-{run_id.lower()}"
    session_attempted = False
    remote_metadata: dict[str, Any] = {}
    error: str | None = None
    stop_status: int | None = None
    exec_status: int | None = None

    try:
        session_attempted = True
        provision_status = stream_command([colab, "new", "-s", session, "--gpu", args.gpu], colab_log)
        if provision_status:
            raise RuntimeError(
                f"Colab could not provision the requested GPU {args.gpu}; no fallback accelerator was selected."
            )
        upload_status = upload_archive(colab, session, input_archive, run_dir, colab_log)
        input_archive.unlink(missing_ok=True)
        if upload_status:
            raise RuntimeError("Colab failed to stage OCRKit training inputs")
        exec_status = stream_command(
            [colab, "exec", "-s", session, "--timeout", str(args.timeout_seconds), "-f", str(REMOTE_RUNNER)],
            colab_log,
        )
        remote_metadata, remote_log_path = fetch_remote_metadata(colab, session, run_dir, colab_log)
        retrieve_checkpoint(r2, remote_metadata, remote_log_path, run_dir)
        if exec_status:
            raise RuntimeError(f"Colab training failed with exit status {exec_status}.")
        if remote_metadata.get("status") != "success":
            raise RuntimeError(
                f"Colab training did not return a successful status: {remote_metadata.get('error', 'unknown error')}"
            )
    except KeyboardInterrupt:
        error = "Colab run interrupted by the operator."
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        input_archive.unlink(missing_ok=True)
        if session_attempted:
            try:
                stop_status = stream_command([colab, "stop", "-s", session], colab_log)
            except Exception as exc:
                stop_status = -1
                stop_error = f"Colab runtime teardown command failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {stop_error}" if error else stop_error
            if stop_status and not session_still_active(colab, session, colab_log):
                # Colab had already released the runtime on its own; nothing was left running.
                stop_status = 0
            if stop_status and error is None:
                error = f"training completed, but Colab runtime teardown failed; run colab stop -s {session}"

    accepted = run_dir / ACCEPTED_STAGING
    if error is None and stop_status in (None, 0) and accepted.is_dir():
        try:
            evaluate_locally(accepted, colab_log, stream_command)
        except KeyboardInterrupt:
            error = "Local checkpoint evaluation interrupted by the operator."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    succeeded = error is None and stop_status in (None, 0)
    if accepted.is_dir():
        if succeeded:
            for child in accepted.iterdir():
                child.rename(run_dir / child.name)
            accepted.rmdir()
        else:
            partial = run_dir / "partial"
            partial.mkdir(exist_ok=True)
            for child in accepted.iterdir():
                child.rename(partial / child.name)
            accepted.rmdir()
    status = {
        "status": "success" if succeeded else "failed",
        "runtime_stopped": stop_status == 0,
        "colab_session": session,
        "error": error,
    }
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if status["status"] != "success":
        print(f"Colab OCRKit run failed: {error or 'runtime did not stop successfully'}", file=sys.stderr)
        print(f"Local run logs and any partial outputs: {run_dir}", file=sys.stderr)
        return 1
    print(f"Colab OCRKit run completed. Checkpoint, local evaluation, provenance, and logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
