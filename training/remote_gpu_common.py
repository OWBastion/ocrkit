#!/usr/bin/env python3
"""Shared local-side contract for remote-GPU training runners (Colab, Kaggle, ...).

This module runs only on the operator's own machine. It owns the parts of a
remote run that do not depend on which backend executes the training: staging
the reviewed dataset/checkpoint/source files into one input archive, verifying
and retrieving the trained checkpoint through R2, and running the unchanged
local evaluation gate. Provisioning, submission, and status/output retrieval
are backend-specific and stay in each runner (`run_rec_colab.py`,
`run_rec_kaggle.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

Runner = Callable[[list[str], Path], int]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Running a runner directly (python training/run_rec_*.py) puts training/, not the
    # repo root, on sys.path; the repo's own `app` package needs the root added explicitly.
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.storage.r2_client import ObjectNotFoundError, R2ObjectStore  # noqa: E402

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
R2_UPLOAD_URL_BUFFER_SECONDS = 900
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
    """Every backend retrieves the checkpoint through R2 rather than its own (slower) transfer."""
    if not (
        settings.r2_endpoint_url
        and settings.r2_access_key_id
        and settings.r2_secret_access_key
        and settings.r2_default_bucket
    ):
        parser.error(
            "OCRKIT_R2_ENDPOINT_URL, OCRKIT_R2_ACCESS_KEY_ID, OCRKIT_R2_SECRET_ACCESS_KEY, and "
            "OCRKIT_R2_DEFAULT_BUCKET are required to run remote GPU training"
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
        raise ValueError(f"required remote-run input is missing or is a symbolic link: {source}")
    if archive_path in files:
        if files[archive_path][0] != source:
            raise ValueError(f"two remote-run inputs map to the same path: {archive_path}")
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


def _collect_stage_files(
    dataset_root: Path,
    checkpoint_path: Path | None,
    train_images: list[str],
    holdout_images: list[str],
) -> dict[str, tuple[Path, str]]:
    """Every backend stages this identical set of files; only the transport differs."""
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
    return files


def _stage_records(files: dict[str, tuple[Path, str]]) -> list[dict[str, Any]]:
    return [
        {"path": path, "kind": kind, "size_bytes": source.stat().st_size, "sha256": sha256(source)}
        for path, (source, kind) in sorted(files.items())
    ]


def stage_inputs(
    archive_path: Path,
    run_request: dict[str, Any],
    dataset_root: Path,
    checkpoint_path: Path | None,
    train_images: list[str],
    holdout_images: list[str],
) -> None:
    """Build the one input archive (+ request.json) every backend stages identically.

    Colab uploads this archive directly through its CLI's chunked transfer; Kaggle instead
    uploads it to R2 and has its kernel download it through a bounded, single-object presigned
    URL (see upload_via_presigned_url below) — Kaggle's own dataset-attachment mechanism silently
    auto-extracts or drops archives/subdirectories depending on undocumented, unstable behavior,
    so it is not used for input transport at all.
    """
    files = _collect_stage_files(dataset_root, checkpoint_path, train_images, holdout_images)
    records = _stage_records(files)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, (source, _kind) in sorted(files.items()):
            archive.add(source, arcname=path, recursive=False)
        request = {"run": run_request, "input_files": records}
        request_path = archive_path.parent / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        archive.add(request_path, arcname="request.json", recursive=False)


def upload_via_presigned_url(path: Path, url: str) -> None:
    """PUTs a local file straight to R2 through a bounded, single-object presigned URL."""
    result = subprocess.run(
        [
            "curl",
            "-fsS",
            "-X",
            "PUT",
            "--data-binary",
            f"@{path}",
            "-H",
            "Content-Type: application/octet-stream",
            "--max-time",
            "1800",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stdout.strip() not in {"200", "201"}:
        raise RuntimeError(f"uploading to R2 failed (curl exit {result.returncode}: {result.stderr.strip()[-300:]})")


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


def transfer_with_retry(command: list[str], log_path: Path, run: Runner, attempts: int = 3) -> int:
    """`run` is the caller's own command runner so a test can patch it in the caller's module."""
    status = 1
    for _ in range(attempts):
        status = run(command, log_path)
        if not status:
            break
    return status


def retrieve_checkpoint(
    r2: R2ObjectStore, remote_metadata: dict[str, Any], remote_log_path: Path | None, run_dir: Path
) -> None:
    """The checkpoint travels through R2 (uploaded by the remote worker script), never the backend CLI."""
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
                f"remote worker did not upload the checkpoint to R2: {checkpoint['bucket']}/{checkpoint['key']}"
            ) from exc
        if destination.stat().st_size != checkpoint["size_bytes"] or sha256(destination) != checkpoint["sha256"]:
            raise ValueError("remote checkpoint failed checksum verification after download from R2")
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
                f"warning: failed to delete the remote checkpoint object from R2 "
                f"({checkpoint['bucket']}/{checkpoint['key']}): {exc}",
                file=sys.stderr,
            )


def evaluate_locally(accepted: Path, log_path: Path, run: Runner) -> None:
    """Run the existing local evaluation contract on the retrieved checkpoint.

    `run` is the caller's own command runner so a test can patch it in the caller's module.
    """
    evaluation = accepted / "evaluation"
    status = run(
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
