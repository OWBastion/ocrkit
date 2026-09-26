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
RUNS = ROOT / "training/.work/colab-runs"
ACCEPTED_STAGING = "accepted"
REMOTE_RUNNER = ROOT / "training/colab_remote.py"
PRETRAINED_CHECKPOINT = ROOT / "training/.work/pretrained/PP-OCRv6_small_rec_pretrained.pdparams"
SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/batch_eval.py",
    "training/bootstrap.sh",
    "training/setup_rec_environment.sh",
    "training/run_rec_smoke.sh",
    "training/evaluate_rec_checkpoint.sh",
    "training/configs/rec_pp_ocrv6_small.yaml",
    "training/configs/pp_ocrv6_small_det.lock.json",
    "training/scripts/prepare_detector.py",
    "training/scripts/prepare_rapidocr_config.py",
    "training/scripts/prune_rec_checkpoints.py",
    "training/scripts/validate_annotations.py",
    "training/colab_remote.py",
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


def add_fixture_set(files: dict[str, tuple[Path, str]], relative_cases: str) -> None:
    cases_path = ROOT / relative_cases
    add_file(files, f"repo/{relative_cases}", cases_path, "evaluation-fixture")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases:
        image_path = safe_relative(case["image"])
        source = cases_path.parent / image_path
        add_file(
            files,
            f"repo/{cases_path.parent.relative_to(ROOT).as_posix()}/{image_path.as_posix()}",
            source,
            "evaluation-fixture",
        )


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
    for directory in ("app", "configs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                add_file(files, f"repo/{path.relative_to(root).as_posix()}", path, "ocrkit-source")
    for relative in SOURCE_FILES:
        add_file(files, f"repo/{relative}", root / relative, "ocrkit-source")
    add_fixture_set(files, "datasets/fixtures/challenge/cases.json")
    add_fixture_set(files, "tests/fixtures/run_code/cases.json")


def stage_inputs(
    archive_path: Path,
    run_request: dict[str, Any],
    dataset_root: Path,
    checkpoint_path: Path,
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


def stream_colab(command: list[str], log_path: Path) -> int:
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


def safe_extract_result(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError("Colab result archive contains an unsupported path or file type")
        archive.extractall(destination)


def copy_remote_result(remote_root: Path, run_dir: Path, *, success: bool) -> dict[str, Any]:
    results = remote_root / "results"
    metadata_path = results / "run.json"
    remote_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    remote_log = results / "remote.log"
    if success:
        if remote_metadata is None or remote_metadata.get("status") != "success":
            remote_error = (remote_metadata or {}).get("error", "no run metadata")
            raise ValueError(f"Colab run did not succeed: {remote_error}")
        for name in ("checkpoint", "evaluation"):
            if not (results / name).is_dir():
                raise ValueError(f"Colab did not return the {name} artifacts")
        output_records = remote_metadata.get("outputs")
        required_outputs = {
            "checkpoint/best_accuracy.pdparams",
            "evaluation/fixture_report.json",
        }
        if not isinstance(output_records, list) or not required_outputs.issubset(
            {record.get("path") for record in output_records}
        ):
            raise ValueError("Colab did not checksum the required checkpoint and evaluation report")
        for record in output_records:
            output = results / safe_relative(record["path"])
            if (
                not output.is_file()
                or output.stat().st_size != record["size_bytes"]
                or sha256(output) != record["sha256"]
            ):
                raise ValueError(f"Colab result failed checksum verification: {record['path']}")
        if not (results / "checkpoint/best_accuracy.pdparams").is_file():
            raise ValueError("Colab result is missing the best-accuracy checkpoint")
        report_path = results / "evaluation/fixture_report.json"
        if not report_path.is_file():
            raise ValueError("Colab result is missing the fixture evaluation report")
        accepted = run_dir / ACCEPTED_STAGING
        for name in ("checkpoint", "evaluation"):
            shutil.copytree(results / name, accepted / name)
        shutil.copy2(metadata_path, accepted / "run.json")
        if remote_log.is_file():
            shutil.copy2(remote_log, accepted / "remote.log")
        return remote_metadata

    partial = run_dir / "partial"
    partial.mkdir(exist_ok=True)
    for name in ("checkpoint", "evaluation"):
        source = results / name
        if source.is_dir():
            shutil.copytree(source, partial / name)
    if metadata_path.is_file():
        shutil.copy2(metadata_path, partial / "run.json")
    if remote_log.is_file():
        shutil.copy2(remote_log, partial / "remote.log")
    return remote_metadata or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OCRKit recognition training and evaluation workflow on Colab GPU.")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "datasets/labeled/rec")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=PRETRAINED_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", default="L4", help="Colab GPU preference; no accelerator fallback is attempted")
    parser.add_argument("--timeout-seconds", type=float, default=6 * 3600, help="Upper bound for the remote training and evaluation run")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    colab = shutil.which("colab")
    if not colab:
        parser.error("Google Colab CLI is required; install it with uv tool install google-colab-cli")

    dataset_root = args.labels_dir.resolve()
    train_label = dataset_root / "labels/train.txt"
    holdout_label = dataset_root / "labels/holdout.txt"
    for path in (train_label, holdout_label, args.pretrained_checkpoint):
        if not path.is_file():
            parser.error(f"required training input is missing: {path}")
    train_count, train_images = validate_labels(train_label)
    holdout_count, holdout_images = validate_labels(holdout_label)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid():x}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    input_archive = run_dir / "ocrkit-input.tar.gz"
    result_archive = run_dir / "ocrkit-result.tar.gz"
    colab_log = run_dir / "colab.log"
    retrieved = run_dir / "retrieved"

    source_revision, source_dirty = git_metadata(ROOT)
    dataset_revision, dataset_dirty = git_metadata(dataset_root)
    try:
        dataset_source = dataset_root.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_source = "external"
    checkpoint_path = args.pretrained_checkpoint.resolve()
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
        "base_checkpoint": {
            "model": "PP-OCRv6_small_rec_pretrained",
            "source_name": checkpoint_path.name,
            "sha256": sha256(checkpoint_path),
        },
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
        provision_status = stream_colab([colab, "new", "-s", session, "--gpu", args.gpu], colab_log)
        if provision_status:
            raise RuntimeError(
                f"Colab could not provision the requested GPU {args.gpu}; no fallback accelerator was selected."
            )
        upload_status = stream_colab(
            [colab, "upload", "-s", session, str(input_archive), "/content/ocrkit-input.tar.gz"],
            colab_log,
        )
        input_archive.unlink(missing_ok=True)
        if upload_status:
            raise RuntimeError("Colab failed to stage OCRKit training inputs")
        exec_status = stream_colab(
            [colab, "exec", "-s", session, "--timeout", str(args.timeout_seconds), "-f", str(REMOTE_RUNNER)],
            colab_log,
        )
        download_status = stream_colab(
            [colab, "download", "-s", session, "/content/ocrkit-result.tar.gz", str(result_archive)],
            colab_log,
        )
        if download_status:
            raise RuntimeError(
                "Colab failed to retrieve the remote run logs and artifacts"
                + (f" after the remote run exited with status {exec_status}" if exec_status else "")
            )
        safe_extract_result(result_archive, retrieved)
        try:
            remote_metadata = copy_remote_result(retrieved, run_dir, success=exec_status == 0)
        except Exception:
            shutil.rmtree(run_dir / ACCEPTED_STAGING, ignore_errors=True)
            try:
                copy_remote_result(retrieved, run_dir, success=False)
            except Exception:
                pass
            raise
        if exec_status:
            raise RuntimeError(f"Colab training or evaluation failed with exit status {exec_status}.")
        if remote_metadata.get("status") != "success":
            raise RuntimeError("Colab training or evaluation did not return a successful status.")
    except KeyboardInterrupt:
        error = "Colab run interrupted by the operator."
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        input_archive.unlink(missing_ok=True)
        if session_attempted:
            try:
                stop_status = stream_colab([colab, "stop", "-s", session], colab_log)
            except Exception as exc:
                stop_status = -1
                stop_error = f"Colab runtime teardown command failed: {type(exc).__name__}: {exc}"
                error = f"{error}; {stop_error}" if error else stop_error
            if stop_status and error is None:
                error = f"training completed, but Colab runtime teardown failed; run colab stop -s {session}"

    succeeded = error is None and stop_status in (None, 0)
    accepted = run_dir / ACCEPTED_STAGING
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
    result_archive.unlink(missing_ok=True)
    if retrieved.exists():
        shutil.rmtree(retrieved)
    if status["status"] != "success":
        print(f"Colab OCRKit run failed: {error or 'runtime did not stop successfully'}", file=sys.stderr)
        print(f"Local run logs and any partial outputs: {run_dir}", file=sys.stderr)
        return 1
    print(f"Colab OCRKit run completed. Checkpoint, evaluation, provenance, and logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
