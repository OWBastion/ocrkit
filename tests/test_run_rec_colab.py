from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.storage.r2_client import ObjectNotFoundError
from training import run_rec_colab

SESSION_STOP = "stop"


class FakeR2Store:
    """Stands in for R2ObjectStore: an in-memory object map keyed by (bucket, key)."""

    def __init__(self) -> None:
        self.default_bucket = "test-bucket"
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []

    def generate_presigned_put_url(self, bucket: str, key: str, expires_in_seconds: int) -> str:
        return f"https://example.invalid/put/{bucket}/{key}?expires={expires_in_seconds}"

    def download_object(self, bucket: str, key: str, destination: Path) -> None:
        data = self.objects.get((bucket, key))
        if data is None:
            raise ObjectNotFoundError(f"no such object: {bucket}/{key}")
        destination.write_bytes(data)

    def delete_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)
        self.deleted.append((bucket, key))


@pytest.fixture
def colab_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        "exec_status": 0,
        "remote_status": "success",
        "stop_status": 0,
        "eval_status": 0,
        "upload_checkpoint": True,
        "run_request": None,
        "remote_log_text": "remote log contents\n",
        "checkpoint_config": "Global: {}\n",
        "train_log": "epoch 1 done\n",
        "metadata_download_status": 0,
        "session_active_after_stop_failure": False,
    }
    fake_r2 = FakeR2Store()

    def fake_capture(command: list[str]) -> SimpleNamespace:
        assert command[1] == "sessions"
        if behavior["session_active_after_stop_failure"]:
            session_name = next(c[3] for c in calls if len(c) > 1 and c[1] == "new")
            return SimpleNamespace(returncode=0, stdout=f"[{session_name}] fake-endpoint | Hardware: T4\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.\n", stderr="")

    def fake_stream(command: list[str], _log: Path) -> int:
        calls.append(command)
        if command[0].endswith("evaluate_rec_checkpoint.sh"):
            if behavior["eval_status"]:
                return behavior["eval_status"]
            output = Path(command[-1])
            output.mkdir(parents=True)
            (output / "fixture_report.json").write_text('{"field_accuracy": 0.99, "run_code": {"field_accuracy": 1.0}}')
            return 0
        verb = command[1]
        if verb == "exec":
            # Stands in for colab_remote.py actually running: populate the fake R2 object and
            # the small metadata payload that fetch_remote_metadata/retrieve_checkpoint consume.
            run_request = behavior["run_request"]
            metadata: dict[str, object] = {"status": behavior["remote_status"]}
            if behavior["remote_status"] != "success":
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
            behavior["remote_metadata"] = metadata
            return behavior["exec_status"]
        if verb == "download":
            remote_path, local_path = command[4], Path(command[-1])
            if remote_path.endswith("run.json"):
                if behavior["metadata_download_status"]:
                    return behavior["metadata_download_status"]
                local_path.write_text(json.dumps(behavior["remote_metadata"]), encoding="utf-8")
                return 0
            if remote_path.endswith("remote.log"):
                local_path.write_text(behavior["remote_log_text"], encoding="utf-8")
                return 0
            raise AssertionError(f"unexpected download path: {remote_path}")
        if verb == SESSION_STOP:
            return behavior["stop_status"]
        return 0

    monkeypatch.setattr(run_rec_colab, "RUNS", runs)
    monkeypatch.setattr(run_rec_colab.shutil, "which", lambda _name: "colab")
    monkeypatch.setattr(run_rec_colab, "validate_labels", lambda _path: (1, ["a.png"]))
    monkeypatch.setattr(run_rec_colab, "PART_BYTES", 4)
    monkeypatch.setattr(run_rec_colab, "require_r2_store", lambda _parser: fake_r2)

    def fake_stage(archive: Path, run_request: dict, *_rest) -> None:
        behavior["run_request"] = run_request
        archive.write_bytes(b"input-archive")

    monkeypatch.setattr(run_rec_colab, "stage_inputs", fake_stage)
    monkeypatch.setattr(run_rec_colab, "stream_command", fake_stream)
    monkeypatch.setattr(run_rec_colab, "capture_command", fake_capture)
    monkeypatch.setattr(
        sys, "argv", ["run_rec_colab.py", "--labels-dir", str(labels), "--pretrained-checkpoint", str(checkpoint)]
    )
    return runs, calls, behavior, fake_r2


def only_run(runs: Path) -> Path:
    (run_dir,) = runs.iterdir()
    return run_dir


def test_training_failure_keeps_partial_checkpoint_and_stops_runtime(colab_run) -> None:
    runs, calls, behavior, fake_r2 = colab_run
    behavior.update(exec_status=1, remote_status="failed")

    assert run_rec_colab.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert calls[-1][1] == SESSION_STOP
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert not fake_r2.objects  # still deleted even though the run overall failed


def test_local_evaluation_failure_demotes_checkpoint_to_partial(colab_run) -> None:
    runs, _calls, behavior, _fake_r2 = colab_run
    behavior["eval_status"] = 1

    assert run_rec_colab.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert "local checkpoint evaluation failed" in json.loads((run_dir / "status.json").read_text())["error"]
    assert json.loads((run_dir / "status.json").read_text())["runtime_stopped"] is True


def test_tampered_checkpoint_is_rejected_and_still_deleted(tmp_path: Path) -> None:
    fake_r2 = FakeR2Store()
    fake_r2.objects[("bucket", "key")] = b"tampered-bytes"
    remote_metadata = {
        "status": "success",
        "checkpoint": {"bucket": "bucket", "key": "key", "size_bytes": 999, "sha256": "0" * 64},
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="checksum"):
        run_rec_colab.retrieve_checkpoint(fake_r2, remote_metadata, None, run_dir)

    assert not fake_r2.objects  # cleaned up even though verification failed
    assert json.loads((run_dir / "accepted" / "run.json").read_text())["status"] == "success"
