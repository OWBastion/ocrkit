#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Running this file directly (python training/run_rec_colab.py) puts training/, not the
    # repo root, on sys.path; the repo's own `app` package needs the root added explicitly.
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.storage.r2_client import ObjectNotFoundError, R2ObjectStore  # noqa: E402
RUNS = ROOT / "training/.work/colab-runs"
ACCEPTED_STAGING = "accepted"
PADDLE_WHEEL = {
    "url": "https://cdn.owbastion.codes/ocrkit/wheels/cu129/paddlepaddle_gpu-3.3.1-cp312-cp312-linux_x86_64.whl",
    "sha256": "03fc5211183ba20ef71e63a35e589fd386a8805c1011c3a409a0e5b118be4668",
}
OFFICIAL_BASE_CHECKPOINT = {
    "model": "PP-OCRv6_small_rec_pretrained",
    "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams",
    "sha256": "25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825",
}
R2_KEY_PREFIX = "colab-runs"
R2_UPLOAD_URL_BUFFER_SECONDS = 900
PART_BYTES = 32 * 1024 * 1024
REMOTE_RUNNER = ROOT / "training/colab_remote.py"
PRETRAINED_CHECKPOINT = ROOT / "training/.work/pretrained/PP-OCRv6_small_rec_pretrained.pdparams"
SOURCE_FILES = (
    "training/bootstrap.sh",
    "training/setup_rec_environment.sh",
    "training/run_rec_smoke.sh",
    "training/configs/rec_pp_ocrv6_small.yaml",
    "training/scripts/prune_rec_checkpoints.py",
    "training/scripts/validate_annotations.py",
)


def require_r2_store(parser: argparse.ArgumentParser) -> R2ObjectStore:
    """The Colab backend retrieves checkpoints through R2 rather than the slow Colab CLI transfer."""
    if not (
        settings.r2_endpoint_url
        and settings.r2_access_key_id
        and settings.r2_secret_access_key
        and settings.r2_default_bucket
    ):
        parser.error(
            "OCRKIT_R2_ENDPOINT_URL, OCRKIT_R2_ACCESS_KEY_ID, OCRKIT_R2_SECRET_ACCESS_KEY, and "
            "OCRKIT_R2_DEFAULT_BUCKET are required to run training on Colab"
        )
    return R2ObjectStore.from_settings(
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region_name,
        default_bucket=settings.r2_default_bucket,
        allowed_buckets_raw=settings.r2_allowed_buckets,
        read_timeout_seconds=settings.r2_read_timeout_seconds,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(files: dict[str, tuple[Path, str]], archive_path: str, source: Path, kind: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"required Colab input is missing or is a symbolic link: {source}")
    if archive_path in files:
        if files[archive_path][0] != source:
            raise ValueError(f"two Colab inputs map to the same path: {archive_path}")
        return
    files[archive_path] = (source, kind)


def safe_relative(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in PurePosixPath(value).parts)
    ):
        raise ValueError(f"training input contains an unsafe image path: {value!r}")
    return path


def validate_labels(label_path: Path) -> tuple[int, list[str]]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "training/scripts/validate_annotations.py"), "rec", str(label_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sample_count = int(json.loads(result.stdout)["valid_samples"])
    if sample_count < 1:
        raise ValueError(f"recognition split is empty: {label_path.name}")
    images = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_name = line.split("\t", 1)[0]
        relative = safe_relative(image_name)
        candidates = (label_path.parent / "images" / relative, label_path.parent.parent / relative)
        image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if image_path is None:
            raise ValueError(f"validated recognition image disappeared: {image_name}")
        image_path = image_path.resolve()
        dataset_root = label_path.parent.parent.resolve()
        if not image_path.is_relative_to(dataset_root):
            raise ValueError(f"recognition image resolves outside the selected dataset: {image_name}")
        images.append(image_name)
    return sample_count, images


def git_metadata(path: Path) -> tuple[str | None, bool]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode:
        return None, False
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return revision.stdout.strip(), bool(status.stdout.strip())


def read_dataset_provenance(dataset_root: Path) -> dict[str, Any]:
    for name in ("provenance.json", "snapshot.json"):
        path = dataset_root / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        snapshot = data.get("snapshot", {})
        snapshot_id = snapshot.get("snapshot_id") or data.get("snapshot_id")
        if snapshot_id:
            return {
                "snapshot_id": snapshot_id,
                "snapshot_version": snapshot.get("version") or data.get("snapshot_version") or data.get("version"),
                "code_revision": data.get("code_revision"),
            }
    return {}


def source_files(root: Path, files: dict[str, tuple[Path, str]]) -> None:
    for relative in SOURCE_FILES:
        add_file(files, f"repo/{relative}", root / relative, "ocrkit-source")


def stage_inputs(
    archive_path: Path,
    run_request: dict[str, Any],
    dataset_root: Path,
    checkpoint_path: Path | None,
    train_images: list[str],
    holdout_images: list[str],
) -> None:
    files: dict[str, tuple[Path, str]] = {}
    source_files(ROOT, files)

    for relative in ("labels/train.txt", "labels/holdout.txt"):
        add_file(files, f"dataset/{relative}", dataset_root / relative, "reviewed-labels")
    image_names = set(train_images + holdout_images)
    for image_name in sorted(image_names):
        relative = safe_relative(image_name)
        label_parent = dataset_root / "labels"
        candidates = (label_parent / "images" / relative, dataset_root / relative)
        image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if image_path is None:
            raise ValueError(f"training crop is missing: {image_name}")
        add_file(files, f"dataset/{relative.as_posix()}", image_path, "reviewed-crop")

    for relative in (
        "provenance.json",
        "snapshot.json",
        "crop_manifest.json",
        "review/train.jsonl",
        "review/holdout.jsonl",
    ):
        path = dataset_root / relative
        if path.is_file():
            add_file(files, f"dataset/{relative}", path, "dataset-provenance")
    if checkpoint_path is not None:
        add_file(
            files,
            "repo/training/.work/pretrained/PP-OCRv6_small_rec_pretrained.pdparams",
            checkpoint_path,
            "base-recognition-checkpoint",
        )

    records = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, (source, kind) in sorted(files.items()):
            archive.add(source, arcname=path, recursive=False)
            records.append(
                {
                    "path": path,
                    "kind": kind,
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256(source),
                }
            )
        request = {"run": run_request, "input_files": records}
        request_path = archive_path.parent / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        archive.add(request_path, arcname="request.json", recursive=False)


def stream_command(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        rendered = shlex.join(command)
        print(f"$ {rendered}", flush=True)
        log.write(f"$ {rendered}\n")
        with subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            return process.wait()


def capture_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def session_still_active(colab: str, session: str, log_path: Path) -> bool:
    """`colab stop` can fail simply because Colab already reclaimed an idle/finished runtime."""
    result = capture_command([colab, "sessions"])
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {colab} sessions\n{result.stdout}{result.stderr}\n")
    if result.returncode:
        return True  # could not verify; assume active so a real leak is not silently dropped
    return session in result.stdout


def transfer_with_retry(command: list[str], log_path: Path, attempts: int = 3) -> int:
    status = 1
    for _ in range(attempts):
        status = stream_command(command, log_path)
        if not status:
            break
    return status


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
        [colab, "download", "-s", session, "/content/ocrkit-colab/results/run.json", str(metadata_path)], log_path
    )
    if status:
        raise RuntimeError("Colab did not return run metadata (results/run.json)")
    remote_log_path = run_dir / "remote-log.txt"
    if transfer_with_retry(
        [colab, "download", "-s", session, "/content/ocrkit-colab/results/remote.log", str(remote_log_path)], log_path
    ):
        remote_log_path = None  # best-effort: diagnostics, not required for the run's outcome
    return json.loads(metadata_path.read_text(encoding="utf-8")), remote_log_path


def retrieve_checkpoint(
    r2: R2ObjectStore, remote_metadata: dict[str, Any], remote_log_path: Path | None, run_dir: Path
) -> None:
    """The checkpoint travels through R2 (uploaded by colab_remote.py), not the Colab CLI."""
    accepted = run_dir / ACCEPTED_STAGING
    accepted.mkdir(parents=True, exist_ok=True)
    (accepted / "run.json").write_text(
        json.dumps(remote_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if remote_log_path is not None:
        shutil.move(str(remote_log_path), accepted / "remote.log")

    checkpoint = remote_metadata.get("checkpoint")
    if checkpoint is None:
        return
    checkpoint_dir = accepted / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / "best_accuracy.pdparams"
    try:
        try:
            r2.download_object(checkpoint["bucket"], checkpoint["key"], destination)
        except ObjectNotFoundError as exc:
            raise ValueError(
                f"Colab did not upload the checkpoint to R2: {checkpoint['bucket']}/{checkpoint['key']}"
            ) from exc
        if destination.stat().st_size != checkpoint["size_bytes"] or sha256(destination) != checkpoint["sha256"]:
            raise ValueError("Colab checkpoint failed checksum verification after download from R2")
        config_text = remote_metadata.get("checkpoint_config")
        if config_text:
            (checkpoint_dir / "config.yml").write_text(config_text, encoding="utf-8")
        train_log_text = remote_metadata.get("train_log")
        if train_log_text:
            (checkpoint_dir / "train.log").write_text(train_log_text, encoding="utf-8")
    finally:
        try:
            r2.delete_object(checkpoint["bucket"], checkpoint["key"])
        except Exception as exc:
            print(
                f"warning: failed to delete the Colab checkpoint object from R2 "
                f"({checkpoint['bucket']}/{checkpoint['key']}): {exc}",
                file=sys.stderr,
            )


def evaluate_locally(accepted: Path, log_path: Path) -> None:
    """Run the existing local evaluation contract on the retrieved checkpoint."""
    evaluation = accepted / "evaluation"
    status = stream_command(
        [
            str(ROOT / "training/evaluate_rec_checkpoint.sh"),
            str((accepted / "checkpoint/best_accuracy").resolve()),
            str(evaluation.resolve()),
        ],
        log_path,
    )
    if status:
        raise RuntimeError(f"local checkpoint evaluation failed with exit status {status}")
    report = json.loads((evaluation / "fixture_report.json").read_text(encoding="utf-8"))
    metadata = json.loads((accepted / "run.json").read_text(encoding="utf-8"))
    metadata["evaluation"] = {
        "location": "local",
        "report": "evaluation/fixture_report.json",
        "field_accuracy": report.get("field_accuracy"),
        "run_code_accuracy": report.get("run_code", {}).get("field_accuracy"),
    }
    (accepted / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            evaluate_locally(accepted, colab_log)
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
