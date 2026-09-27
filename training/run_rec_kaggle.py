#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    # Running this file directly (python training/run_rec_kaggle.py) puts training/, not the
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
    validate_labels,
)

RUNS = ROOT / "training/.work/kaggle-runs"
R2_KEY_PREFIX = "kaggle-runs"
REMOTE_WORKER = ROOT / "training/kaggle_remote.py"
INPUT_ARCHIVE_NAME = "ocrkit-input.tar.gz"
STATUS_POLL_SECONDS = 20
DATASET_READY_POLL_SECONDS = 5
DATASET_READY_MAX_ATTEMPTS = 30


def _read_json_username(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    username = data.get("username")
    return str(username) if username else None


def resolve_kaggle_username(parser: argparse.ArgumentParser) -> str:
    """Kaggle dataset/kernel ids must be prefixed by the authenticated account's own username."""
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    # The legacy `kaggle.json` API-key file and the OAuth `kaggle auth login` session
    # (`credentials.json`) both record the account's own username under the same key.
    for name in ("kaggle.json", "credentials.json"):
        username = _read_json_username(config_dir / name)
        if username:
            return username
    parser.error(
        "Kaggle username is required to name the private dataset/kernel; set KAGGLE_USERNAME, run "
        "`kaggle auth login`, or place ~/.kaggle/kaggle.json"
    )
    raise AssertionError("unreachable")  # parser.error always raises SystemExit


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_input_dataset(kaggle: str, dataset_id: str, archive: Path, run_dir: Path, log_path: Path) -> int:
    """A private, run-scoped Kaggle Dataset stages the same input archive Colab uploads directly."""
    package_dir = run_dir / "dataset-package"
    package_dir.mkdir()
    shutil.copyfile(archive, package_dir / INPUT_ARCHIVE_NAME)
    write_json(
        package_dir / "dataset-metadata.json",
        {"id": dataset_id, "title": dataset_id.split("/", 1)[1], "licenses": [{"name": "unknown"}]},
    )
    # `datasets create` defaults to private; passing -u/--public would be a privacy regression.
    status = stream_command([kaggle, "datasets", "create", "-p", str(package_dir), "-r", "zip"], log_path)
    if status:
        return status
    # The exact "ready"/processing status text is not stabilized across kaggle-api releases, so this
    # wait is best-effort diagnostics on top of `datasets create` itself already blocking on the upload;
    # it only fails closed on an explicit error report.
    for _ in range(DATASET_READY_MAX_ATTEMPTS):
        result = capture_command([kaggle, "datasets", "status", dataset_id])
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {kaggle} datasets status {dataset_id}\n{result.stdout}{result.stderr}\n")
        if "error" in result.stdout.lower():
            return 1
        if not result.returncode:
            break
        time.sleep(DATASET_READY_POLL_SECONDS)
    return 0


def push_kernel(kaggle: str, kernel_id: str, dataset_id: str, run_dir: Path, log_path: Path) -> int:
    package_dir = run_dir / "kernel-package"
    package_dir.mkdir()
    shutil.copyfile(REMOTE_WORKER, package_dir / "kaggle_remote.py")
    write_json(
        package_dir / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_id.split("/", 1)[1],
            "code_file": "kaggle_remote.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            # Required so the pushed kaggle_remote.py can PUT the checkpoint to R2; no platform or
            # release credentials are ever given to Kaggle, only a bounded, single-object write URL.
            "enable_internet": True,
            "dataset_sources": [dataset_id],
        },
    )
    return stream_command([kaggle, "kernels", "push", "-p", str(package_dir), "--accelerator", "gpu"], log_path)


def poll_kernel_status(kaggle: str, kernel_id: str, timeout_seconds: float, log_path: Path) -> str:
    """Kaggle kernel execution is asynchronous, unlike Colab's synchronous `exec`; poll until settled."""
    deadline = time.monotonic() + timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        result = capture_command([kaggle, "kernels", "status", kernel_id])
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {kaggle} kernels status {kernel_id}\n{result.stdout}{result.stderr}\n")
        last_output = (result.stdout + result.stderr).lower()
        if "error" in last_output or "cancel" in last_output:
            return "error"
        if "complete" in last_output:
            return "complete"
        time.sleep(STATUS_POLL_SECONDS)
    raise RuntimeError(f"Kaggle kernel did not settle within {timeout_seconds:.0f}s (last status: {last_output.strip() or 'unknown'})")


def fetch_kernel_output(kaggle: str, kernel_id: str, run_dir: Path, log_path: Path) -> tuple[dict[str, Any], Path | None]:
    """`kaggle kernels output` pulls the whole /kaggle/working tree, including results/run.json and remote.log."""
    output_dir = run_dir / "kernel-output"
    status = stream_command([kaggle, "kernels", "output", kernel_id, "-p", str(output_dir), "-o"], log_path)
    if status:
        raise RuntimeError("Kaggle did not return kernel output")
    metadata_path = next(output_dir.rglob("run.json"), None)
    if metadata_path is None:
        raise RuntimeError("Kaggle kernel output did not include run metadata (results/run.json)")
    remote_log_path = next(output_dir.rglob("remote.log"), None)
    return json.loads(metadata_path.read_text(encoding="utf-8")), remote_log_path


def delete_input_dataset(kaggle: str, dataset_id: str, log_path: Path) -> int:
    """The input dataset is the only persistent Kaggle storage holding reviewed crops; always remove it."""
    return stream_command([kaggle, "datasets", "delete", dataset_id, "-y"], log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the OCRKit recognition model on a Kaggle GPU, then evaluate it locally.")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "datasets/labeled/rec")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=6 * 3600, help="Upper bound for the remote training run")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    kaggle = shutil.which("kaggle")
    if not kaggle:
        parser.error("Kaggle CLI is required; install it with uv tool install kaggle")
    username = resolve_kaggle_username(parser)

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
    input_archive = run_dir / INPUT_ARCHIVE_NAME
    kaggle_log = run_dir / "kaggle.log"

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
            "recipe": "training/configs/rec_pp_ocrv6_small.yaml",
            "paddleocr_recipe": "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml",
        },
    }
    try:
        stage_inputs(input_archive, run_request, dataset_root, checkpoint_path, train_images, holdout_images)
    except Exception as exc:
        input_archive.unlink(missing_ok=True)
        write_json(run_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise

    # Kaggle datasets and kernels share one per-account slug namespace: reusing the same slug for
    # both makes `kernels push` 409 (SaveKernel Conflict) against the just-created dataset.
    run_slug = run_id.lower()
    dataset_id = f"{username}/ocrkit-rec-data-{run_slug}"
    kernel_id = f"{username}/ocrkit-rec-{run_slug}"
    dataset_created = False
    remote_metadata: dict[str, Any] = {}
    error: str | None = None
    cleanup_status: int | None = None

    try:
        dataset_status = create_input_dataset(kaggle, dataset_id, input_archive, run_dir, kaggle_log)
        input_archive.unlink(missing_ok=True)
        if dataset_status:
            raise RuntimeError("Kaggle failed to stage the private OCRKit input dataset")
        dataset_created = True
        push_status = push_kernel(kaggle, kernel_id, dataset_id, run_dir, kaggle_log)
        if push_status:
            raise RuntimeError("Kaggle could not provision/submit the private training kernel")
        kernel_status = poll_kernel_status(kaggle, kernel_id, args.timeout_seconds, kaggle_log)
        remote_metadata, remote_log_path = fetch_kernel_output(kaggle, kernel_id, run_dir, kaggle_log)
        retrieve_checkpoint(r2, remote_metadata, remote_log_path, run_dir)
        if kernel_status != "complete":
            raise RuntimeError(f"Kaggle kernel execution did not complete successfully: {remote_metadata.get('error', 'unknown error')}")
        if remote_metadata.get("status") != "success":
            raise RuntimeError(
                f"Kaggle training did not return a successful status: {remote_metadata.get('error', 'unknown error')}"
            )
    except KeyboardInterrupt:
        error = "Kaggle run interrupted by the operator."
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        input_archive.unlink(missing_ok=True)
        if dataset_created:
            try:
                cleanup_status = delete_input_dataset(kaggle, dataset_id, kaggle_log)
            except Exception as exc:
                cleanup_status = -1
                cleanup_error = f"Kaggle input dataset cleanup failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {cleanup_error}" if error else cleanup_error
            if cleanup_status and error is None:
                error = f"training completed, but deleting the private input dataset failed; run kaggle datasets delete {dataset_id} -y"

    accepted = run_dir / ACCEPTED_STAGING
    if error is None and cleanup_status in (None, 0) and accepted.is_dir():
        try:
            evaluate_locally(accepted, kaggle_log, stream_command)
        except KeyboardInterrupt:
            error = "Local checkpoint evaluation interrupted by the operator."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    succeeded = error is None and cleanup_status in (None, 0)
    if accepted.is_dir():
        destination = run_dir if succeeded else run_dir / "partial"
        destination.mkdir(exist_ok=True)
        for child in accepted.iterdir():
            child.rename(destination / child.name)
        accepted.rmdir()
    status = {
        "status": "success" if succeeded else "failed",
        "input_dataset_deleted": cleanup_status == 0,
        "kaggle_dataset": dataset_id,
        # The Kaggle CLI has no kernel-delete command; the private kernel and its output stay in the
        # operator's own account. Remove it from https://www.kaggle.com/code if it should not be kept.
        "kaggle_kernel": kernel_id,
        "error": error,
    }
    write_json(run_dir / "status.json", status)
    if status["status"] != "success":
        print(f"Kaggle OCRKit run failed: {error or 'kernel did not complete successfully'}", file=sys.stderr)
        print(f"Local run logs and any partial outputs: {run_dir}", file=sys.stderr)
        return 1
    print(f"Kaggle OCRKit run completed. Checkpoint, local evaluation, provenance, and logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
