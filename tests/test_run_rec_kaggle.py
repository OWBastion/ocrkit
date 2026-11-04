from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import run_rec_kaggle


class FakeR2Store:
    """Stands in for R2ObjectStore: an in-memory object map keyed by (bucket, key)."""

    def __init__(self) -> None:
        self.default_bucket = "test-bucket"
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []

    def generate_presigned_put_url(self, bucket: str, key: str, expires_in_seconds: int) -> str:
        return f"https://example.invalid/put/{bucket}/{key}?expires={expires_in_seconds}"

    def generate_presigned_get_url(self, bucket: str, key: str, expires_in_seconds: int) -> str:
        return f"https://example.invalid/get/{bucket}/{key}?expires={expires_in_seconds}"

    def download_object(self, bucket: str, key: str, destination: Path) -> None:
        from app.storage.r2_client import ObjectNotFoundError

        data = self.objects.get((bucket, key))
        if data is None:
            raise ObjectNotFoundError(f"no such object: {bucket}/{key}")
        destination.write_bytes(data)

    def delete_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)
        self.deleted.append((bucket, key))


def verb(command: list[str]) -> tuple[str, str]:
    return command[1], command[2]


@pytest.fixture
def kaggle_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    labels = tmp_path / "labels-dir"
    (labels / "labels").mkdir(parents=True)
    for name in ("train.txt", "holdout.txt"):
        (labels / "labels" / name).write_text("a.png\ta\n", encoding="utf-8")
    checkpoint = tmp_path / "base.pdparams"
    checkpoint.write_bytes(b"base")
    runs = tmp_path / "runs"
    calls: list[list[str]] = []
    checkpoint_bytes = b"weights"
    behavior = {
        "push_status": 0,
        "kernel_status": "complete",
        "output_status": 0,
        "eval_status": 0,
        "upload_checkpoint": True,
        "run_request": None,
        "remote_log_text": "remote log contents\n",
        "checkpoint_config": "Global: {}\n",
        "train_log": "epoch 1 done\n",
    }
    fake_r2 = FakeR2Store()

    def fake_capture(command: list[str]) -> SimpleNamespace:
        group, action = verb(command)
        if group == "kernels" and action == "status":
            text = "kernel run error\n" if behavior["kernel_status"] == "error" else "kernel run complete\n"
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        raise AssertionError(f"unexpected capture command: {command}")

    def fake_stream(command: list[str], _log: Path) -> int:
        calls.append(command)
        if command[0].endswith("evaluate_rec_checkpoint.sh"):
            if behavior["eval_status"]:
                return behavior["eval_status"]
            output = Path(command[2])
            output.mkdir(parents=True)
            (output / "fixture_report.json").write_text('{"field_accuracy": 0.99, "run_code": {"field_accuracy": 1.0}}')
            return 0
        group, action = verb(command)
        if group == "kernels" and action == "push":
            return behavior["push_status"]
        if group == "kernels" and action == "output":
            if behavior["output_status"]:
                return behavior["output_status"]
            output_dir = Path(command[command.index("-p") + 1]) / "ocrkit-run" / "results"
            output_dir.mkdir(parents=True)
            run_request = behavior["run_request"]
            metadata: dict[str, object] = {
                "status": "success" if behavior["kernel_status"] == "complete" else "failed",
            }
            if behavior["kernel_status"] != "complete":
                metadata["error"] = "RuntimeError: training crashed"
            if behavior["upload_checkpoint"]:
                upload = run_request["checkpoint_upload"]
                fake_r2.objects[(upload["bucket"], upload["key"])] = checkpoint_bytes
                metadata["checkpoint"] = {
                    "bucket": upload["bucket"],
                    "key": upload["key"],
                    "size_bytes": len(checkpoint_bytes),
                    "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                }
                metadata["checkpoint_config"] = behavior["checkpoint_config"]
                metadata["train_log"] = behavior["train_log"]
            (output_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
            (output_dir / "remote.log").write_text(behavior["remote_log_text"], encoding="utf-8")
            return 0
        raise AssertionError(f"unexpected stream command: {command}")

    monkeypatch.setattr(run_rec_kaggle, "RUNS", runs)
    monkeypatch.setattr(run_rec_kaggle.shutil, "which", lambda _name: "kaggle")
    monkeypatch.setattr(run_rec_kaggle, "validate_labels", lambda _path: (1, ["a.png"]))
    monkeypatch.setattr(run_rec_kaggle, "require_r2_store", lambda _parser: fake_r2)
    monkeypatch.setattr(run_rec_kaggle, "STATUS_POLL_SECONDS", 0)
    monkeypatch.setenv("KAGGLE_USERNAME", "ocrkit-operator")

    def fake_stage(archive: Path, run_request: dict, *_rest) -> None:
        behavior["run_request"] = run_request
        archive.write_bytes(b"input-archive")

    def fake_upload(_path: Path, _url: str) -> None:
        pass

    monkeypatch.setattr(run_rec_kaggle, "stage_inputs", fake_stage)
    monkeypatch.setattr(run_rec_kaggle, "upload_via_presigned_url", fake_upload)
    monkeypatch.setattr(run_rec_kaggle, "stream_command", fake_stream)
    monkeypatch.setattr(run_rec_kaggle, "capture_command", fake_capture)
    monkeypatch.setattr(
        sys, "argv", ["run_rec_kaggle.py", "--labels-dir", str(labels), "--pretrained-checkpoint", str(checkpoint)]
    )
    return runs, calls, behavior, fake_r2


def only_run(runs: Path) -> Path:
    (run_dir,) = runs.iterdir()
    return run_dir


def test_kernel_failure_keeps_partial_checkpoint_and_deletes_input_object(kaggle_run) -> None:
    runs, calls, behavior, fake_r2 = kaggle_run
    behavior["kernel_status"] = "error"

    assert run_rec_kaggle.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert not fake_r2.objects  # still deleted even though the run overall failed


def test_local_evaluation_failure_demotes_checkpoint_to_partial(kaggle_run) -> None:
    runs, _calls, behavior, _fake_r2 = kaggle_run
    behavior["eval_status"] = 1

    assert run_rec_kaggle.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert "local checkpoint evaluation failed" in json.loads((run_dir / "status.json").read_text())["error"]
