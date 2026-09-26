#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COLAB_ROOT = Path("/content/ocrkit-colab")
CONTENT = Path("/content")
INPUT_ARCHIVE = CONTENT / "ocrkit-input.tar.gz"
RESULT_ARCHIVE = CONTENT / "ocrkit-result.tar.gz"
RESULT_INDEX = CONTENT / "ocrkit-result.index.json"
PART_BYTES = 32 * 1024 * 1024
REPO = COLAB_ROOT / "repo"
DATASET = COLAB_ROOT / "dataset"
RESULTS = COLAB_ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoint"
EVALUATION = RESULTS / "evaluation"
RUN_METADATA = RESULTS / "run.json"
REMOTE_LOG = RESULTS / "remote.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise RuntimeError("Colab input archive contains an unsupported path or file type")
        archive.extractall(destination)


def run_logged(command: list[str], *, cwd: Path, log: Any, env: dict[str, str] | None = None) -> None:
    rendered = shlex.join(command)
    print(f"$ {rendered}", flush=True)
    log.write(f"$ {rendered}\n")
    log.flush()
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
    log.flush()
    if status:
        raise RuntimeError(f"remote command exited with status {status}: {rendered}")


def checked_gpu() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("Colab did not provide an NVIDIA GPU runtime")
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
        raise RuntimeError("Colab could not report an allocated NVIDIA GPU")
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


def output_files() -> list[dict[str, Any]]:
    records = []
    for directory in (CHECKPOINTS, EVALUATION):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                records.append(
                    {
                        "path": path.relative_to(RESULTS).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return records


def join_input_parts() -> None:
    parts = sorted(CONTENT.glob("ocrkit-input.part*"))
    if not parts:
        raise RuntimeError("OCRKit input parts were not uploaded to the Colab runtime")
    with INPUT_ARCHIVE.open("wb") as archive:
        for part in parts:
            archive.write(part.read_bytes())
            part.unlink()


def split_result_archive() -> None:
    for stale in (*CONTENT.glob("ocrkit-result.part*"), RESULT_INDEX):
        stale.unlink(missing_ok=True)
    parts = []
    with RESULT_ARCHIVE.open("rb") as archive:
        for index, chunk in enumerate(iter(lambda: archive.read(PART_BYTES), b"")):
            name = f"ocrkit-result.part{index:04d}"
            (CONTENT / name).write_bytes(chunk)
            parts.append({"name": name, "sha256": hashlib.sha256(chunk).hexdigest()})
    RESULT_INDEX.write_text(json.dumps({"parts": parts}), encoding="utf-8")


def write_result_archive() -> None:
    RESULT_ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(RESULT_ARCHIVE, "w:gz") as archive:
        for path in (RUN_METADATA, REMOTE_LOG):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(COLAB_ROOT))
        for directory in (CHECKPOINTS, EVALUATION):
            if directory.is_dir():
                archive.add(directory, arcname=directory.relative_to(COLAB_ROOT))
    split_result_archive()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    try:
        with REMOTE_LOG.open("w", encoding="utf-8") as log:
            try:
                join_input_parts()
                COLAB_ROOT.mkdir(parents=True, exist_ok=True)
                safe_extract(INPUT_ARCHIVE, COLAB_ROOT)
                request_path = COLAB_ROOT / "request.json"
                request = json.loads(request_path.read_text(encoding="utf-8"))
                result["request"] = request["run"]
                verify_inputs(request, COLAB_ROOT)
                gpu = checked_gpu()
                REPO.mkdir(parents=True, exist_ok=True)
                DATASET.mkdir(parents=True, exist_ok=True)
                CHECKPOINTS.mkdir(parents=True, exist_ok=True)

                if not shutil.which("uv"):
                    run_logged(
                        [sys.executable, "-m", "pip", "install", "uv"],
                        cwd=COLAB_ROOT,
                        log=log,
                    )
                run_logged(
                    ["uv", "sync", "--locked", "--no-dev", "--python", sys.executable],
                    cwd=REPO,
                    log=log,
                )
                environment = os.environ.copy()
                environment["OCRKIT_TRAINING_PYTHON"] = sys.executable
                run_logged(
                    ["bash", "training/setup_rec_environment.sh", "--device", "cuda"],
                    cwd=REPO,
                    env=environment,
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

                epochs = str(request["run"]["epochs"])
                run_logged(
                    [
                        "bash",
                        "training/run_rec_smoke.sh",
                        "--labels-dir",
                        str(DATASET),
                        "--output-dir",
                        str(CHECKPOINTS),
                        "--evaluation-dir",
                        str(EVALUATION),
                        "--epochs",
                        epochs,
                        "--device",
                        "cuda",
                    ],
                    cwd=REPO,
                    log=log,
                )
                best_checkpoint = CHECKPOINTS / "best_accuracy.pdparams"
                report_path = EVALUATION / "fixture_report.json"
                if not best_checkpoint.is_file() or best_checkpoint.stat().st_size == 0:
                    raise RuntimeError("training did not produce the best-accuracy recognition checkpoint")
                if not report_path.is_file():
                    raise RuntimeError("checkpoint evaluation did not produce fixture_report.json")
                evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
                paddleocr_revision = subprocess.run(
                    ["git", "-C", str(REPO / "training/.work/PaddleOCR"), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                result.update(
                    {
                        "status": "success",
                        "runtime": {
                            **gpu,
                            "python": sys.version,
                            "paddle": paddle_info,
                            "paddleocr_revision": paddleocr_revision,
                        },
                        "evaluation": {
                            "report": "evaluation/fixture_report.json",
                            "field_accuracy": evaluation_report.get("field_accuracy"),
                            "run_code_accuracy": evaluation_report.get("run_code", {}).get("field_accuracy"),
                        },
                        "outputs": output_files(),
                    }
                )
            except BaseException as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=log)
                log.flush()
            finally:
                result["completed_at"] = datetime.now(UTC).isoformat()
                result["outputs"] = output_files()
                RUN_METADATA.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        write_result_archive()
    except BaseException as exc:
        print(f"OCRKit Colab run failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        try:
            RUN_METADATA.write_text(
                json.dumps(
                    {"schema_version": 1, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            write_result_archive()
        except OSError:
            pass
        return 1
    if result.get("status") != "success":
        print(f"OCRKit Colab run failed: {result.get('error', 'remote step failed')}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
