#!/usr/bin/env python3
"""Runs inside the Kaggle kernel. Pushed as the kernel's sole `code_file`, so it must stay a
single self-contained stdlib-only script exactly like `colab_remote.py`: Kaggle does not attach
sibling repository files to a script kernel the way `run_rec_kaggle.py` stages the rest of the
OCRKit repo through the attached input dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKING = Path("/kaggle/working")
KAGGLE_ROOT = WORKING / "ocrkit-run"
REPO = KAGGLE_ROOT / "repo"
DATASET = KAGGLE_ROOT / "dataset"
RESULTS = KAGGLE_ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoint"
RUN_METADATA = RESULTS / "run.json"
REMOTE_LOG = RESULTS / "remote.log"
BASE_CHECKPOINT_PATH = REPO / "training/.work/pretrained/PP-OCRv6_small_rec_pretrained.pdparams"
INPUT_ROOT = Path("/kaggle/input")
# `request.json` sits at the root of the staged input tree (see stage_input_directory in
# training/remote_gpu_common.py) and uniquely identifies the run's mounted dataset.
INPUT_MARKER_NAME = "request.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_input_directory() -> Path:
    """The run's private input dataset is the only dataset attached to this kernel.

    Kaggle Datasets keep an uploaded directory tree natively (and auto-extract any archive
    format by content on mount), so the staged files arrive as plain files under
    /kaggle/input/<dataset-slug>/ rather than as a single blob to extract.
    """
    matches = sorted(INPUT_ROOT.glob(f"*/{INPUT_MARKER_NAME}"))
    if not matches:
        raise RuntimeError(f"OCRKit input dataset was not attached to this Kaggle kernel ({INPUT_MARKER_NAME} not found)")
    if len(matches) > 1:
        raise RuntimeError(f"expected exactly one attached OCRKit input dataset, found {len(matches)}")
    return matches[0].parent


def run_logged(command: list[str], *, cwd: Path, log: Any, env: dict[str, str] | None = None) -> None:
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    log.write(f"$ {rendered}\n")
    log.flush()
    started = time.monotonic()
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    elapsed = round(time.monotonic() - started, 1)
    log.write(f"$ {rendered} -> exit {status} in {elapsed}s\n")
    log.flush()
    if status:
        raise RuntimeError(f"remote command exited with status {status}: {rendered}")


def timed(stages: dict[str, float], name: str):
    class _Timer:
        def __enter__(self):
            self._start = time.monotonic()
            return self

        def __exit__(self, *_exc):
            stages[name] = round(time.monotonic() - self._start, 1)

    return _Timer()


def checked_gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("Kaggle did not provide an NVIDIA GPU runtime")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("Kaggle could not report an allocated NVIDIA GPU")
    devices = []
    for line in result.stdout.splitlines():
        name, capability, driver, memory = (part.strip() for part in line.split(",", 3))
        devices.append(
            {
                "name": name,
                "compute_capability": capability,
                "driver_version": driver,
                "memory_mib": int(memory),
            }
        )
    return {"devices": devices, "nvidia_smi": subprocess.run(
        ["nvidia-smi"], check=False, capture_output=True, text=True
    ).stdout}


def verify_inputs(request: dict[str, Any], root: Path) -> None:
    for record in request["input_files"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"staged input failed checksum verification: {record['path']}")


def fetch_base_checkpoint(base_checkpoint: dict[str, Any]) -> None:
    """A checkpoint the runner did not upload; Kaggle downloads it directly instead."""
    BASE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base_checkpoint["url"], BASE_CHECKPOINT_PATH)
    if sha256(BASE_CHECKPOINT_PATH) != base_checkpoint["sha256"]:
        raise RuntimeError("downloaded base checkpoint failed checksum verification")


def upload_checkpoint(upload: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """PUTs the trained checkpoint straight to R2; Kaggle kernel output has no room for it."""
    result = subprocess.run(
        [
            "curl",
            "-fsS",
            "-X",
            "PUT",
            "--data-binary",
            f"@{checkpoint_path}",
            "-H",
            "Content-Type: application/octet-stream",
            "--max-time",
            "1800",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            upload["url"],
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stdout.strip() not in {"200", "201"}:
        raise RuntimeError(f"uploading the checkpoint to R2 failed (curl exit {result.returncode}: {result.stderr.strip()[-300:]})")
    return {
        "bucket": upload["bucket"],
        "key": upload["key"],
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256(checkpoint_path),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stages: dict[str, float] = {}
    result: dict[str, Any] = {
        "schema_version": 2,
        "backend": "kaggle",
        "status": "failed",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    try:
        with REMOTE_LOG.open("w", encoding="utf-8") as log:
            try:
                input_directory = locate_input_directory()
                KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
                shutil.copytree(input_directory, KAGGLE_ROOT, dirs_exist_ok=True)
                request_path = KAGGLE_ROOT / "request.json"
                request = json.loads(request_path.read_text(encoding="utf-8"))
                run_request = request["run"]
                # The presigned checkpoint-upload URL is a write credential; never persist or log it.
                result["request"] = {
                    key: (value if key != "checkpoint_upload" else {k: v for k, v in value.items() if k != "url"})
                    for key, value in run_request.items()
                }
                verify_inputs(request, KAGGLE_ROOT)
                gpu = checked_gpu()
                REPO.mkdir(parents=True, exist_ok=True)
                DATASET.mkdir(parents=True, exist_ok=True)
                CHECKPOINTS.mkdir(parents=True, exist_ok=True)

                if run_request["base_checkpoint"]["source"] == "official-download":
                    with timed(stages, "fetch_base_checkpoint"):
                        fetch_base_checkpoint(run_request["base_checkpoint"])
                elif not BASE_CHECKPOINT_PATH.is_file():
                    raise RuntimeError("uploaded base checkpoint was not staged at the expected path")

                if not shutil.which("uv"):
                    run_logged([sys.executable, "-m", "pip", "install", "uv"], cwd=KAGGLE_ROOT, log=log)
                mirror = run_request["paddle_wheel_mirror"]
                with timed(stages, "setup_environment"):
                    run_logged(
                        ["bash", "training/setup_rec_environment.sh", "--device", "cuda"],
                        cwd=REPO,
                        env={**os.environ, "OCRKIT_PADDLE_WHEEL_URL": mirror["url"], "OCRKIT_PADDLE_WHEEL_SHA256": mirror["sha256"]},
                        log=log,
                    )

                paddle = subprocess.run(
                    [
                        str(REPO / "training/.work/venv/bin/python"),
                        "-c",
                        "import json, paddle; paddle.device.set_device('gpu:0'); paddle.to_tensor([1.0]).numpy(); print(json.dumps({'version': paddle.__version__, 'cuda': paddle.is_compiled_with_cuda(), 'device': paddle.device.get_device()}))",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                paddle_info = json.loads(paddle.stdout.strip().splitlines()[-1])
                if not paddle_info["cuda"] or paddle_info["device"] != "gpu:0":
                    raise RuntimeError("PaddlePaddle did not select the allocated CUDA device")

                epochs = str(run_request["training"]["epochs"])
                with timed(stages, "train"):
                    run_logged(
                        [
                            "bash",
                            "training/run_rec_smoke.sh",
                            "--labels-dir",
                            str(DATASET),
                            "--output-dir",
                            str(CHECKPOINTS),
                            "--epochs",
                            epochs,
                            "--device",
                            "cuda",
                            "--train-only",
                        ],
                        cwd=REPO,
                        log=log,
                    )
                best_checkpoint = CHECKPOINTS / "best_accuracy.pdparams"
                if not best_checkpoint.is_file() or best_checkpoint.stat().st_size == 0:
                    raise RuntimeError("training did not produce the best-accuracy recognition checkpoint")

                with timed(stages, "upload_checkpoint"):
                    checkpoint_record = upload_checkpoint(run_request["checkpoint_upload"], best_checkpoint)
                # Recorded immediately: if a later step fails, the caller still knows this object
                # exists in R2 and can retrieve or delete it instead of leaking it silently.
                result["checkpoint"] = checkpoint_record

                paddleocr_revision = subprocess.run(
                    ["git", "-C", str(REPO / "training/.work/PaddleOCR"), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                config_path = CHECKPOINTS / "config.yml"
                train_log_path = CHECKPOINTS / "train.log"
                result.update(
                    {
                        "status": "success",
                        "runtime": {
                            **gpu,
                            "python": sys.version,
                            "paddle": paddle_info,
                            "paddleocr_revision": paddleocr_revision,
                        },
                        "stage_seconds": stages,
                        "checkpoint": checkpoint_record,
                        "checkpoint_config": config_path.read_text(encoding="utf-8") if config_path.is_file() else None,
                        "train_log": train_log_path.read_text(encoding="utf-8", errors="replace") if train_log_path.is_file() else None,
                    }
                )
            except BaseException as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["stage_seconds"] = stages
                traceback.print_exc(file=log)
                log.flush()
            finally:
                result["completed_at"] = datetime.now(UTC).isoformat()
                RUN_METADATA.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except BaseException as exc:
        print(f"OCRKit Kaggle run failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        try:
            RUN_METADATA.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "backend": "kaggle",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "stage_seconds": stages,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return 1
    if result.get("status") != "success":
        print(f"OCRKit Kaggle run failed: {result.get('error', 'remote step failed')}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
