from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import run_rec_kaggle

DATASETS_DELETE = ("datasets", "delete")


class FakeR2Store:
    """Stands in for R2ObjectStore: an in-memory object map keyed by (bucket, key)."""

    def __init__(self) -> None:
        self.default_bucket = "test-bucket"
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []

    def generate_presigned_put_url(self, bucket: str, key: str, expires_in_seconds: int) -> str:
        return f"https://example.invalid/put/{bucket}/{key}?expires={expires_in_seconds}"

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
        "dataset_status": 0,
        "push_status": 0,
        "kernel_status": "complete",
        "output_status": 0,
        "eval_status": 0,
        "delete_status": 0,
        "upload_checkpoint": True,
        "run_request": None,
        "remote_log_text": "remote log contents\n",
        "checkpoint_config": "Global: {}\n",
        "train_log": "epoch 1 done\n",
    }
    fake_r2 = FakeR2Store()

    def fake_capture(command: list[str]) -> SimpleNamespace:
        group, action = verb(command)
        if group == "datasets" and action == "status":
            return SimpleNamespace(returncode=0, stdout="dataset ready\n", stderr="")
        if group == "kernels" and action == "status":
            text = "kernel run error\n" if behavior["kernel_status"] == "error" else "kernel run complete\n"
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        raise AssertionError(f"unexpected capture command: {command}")

    def fake_stream(command: list[str], _log: Path) -> int:
        calls.append(command)
        if command[0].endswith("evaluate_rec_checkpoint.sh"):
            if behavior["eval_status"]:
                return behavior["eval_status"]
            output = Path(command[-1])
            output.mkdir(parents=True)
            (output / "fixture_report.json").write_text('{"field_accuracy": 0.99, "run_code": {"field_accuracy": 1.0}}')
            return 0
        group, action = verb(command)
        if group == "datasets" and action == "create":
            return behavior["dataset_status"]
        if group == "datasets" and action == "delete":
            return behavior["delete_status"]
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
    monkeypatch.setattr(run_rec_kaggle, "DATASET_READY_POLL_SECONDS", 0)
    monkeypatch.setenv("KAGGLE_USERNAME", "ocrkit-operator")

    def fake_stage(archive: Path, run_request: dict, *_rest) -> None:
        behavior["run_request"] = run_request
        archive.write_bytes(b"input-archive")

    monkeypatch.setattr(run_rec_kaggle, "stage_inputs", fake_stage)
    monkeypatch.setattr(run_rec_kaggle, "stream_command", fake_stream)
    monkeypatch.setattr(run_rec_kaggle, "capture_command", fake_capture)
    monkeypatch.setattr(
        sys, "argv", ["run_rec_kaggle.py", "--labels-dir", str(labels), "--pretrained-checkpoint", str(checkpoint)]
    )
    return runs, calls, behavior, fake_r2


def only_run(runs: Path) -> Path:
    (run_dir,) = runs.iterdir()
    return run_dir


def test_success_retrieves_checkpoint_from_r2_and_deletes_input_dataset(kaggle_run) -> None:
    runs, calls, behavior, fake_r2 = kaggle_run

    assert run_rec_kaggle.main() == 0

    request = behavior["run_request"]
    assert request["training"]["epochs"] == 10
    assert request["checkpoint_upload"]["bucket"] == fake_r2.default_bucket

    create_call = next(c for c in calls if verb(c) == ("datasets", "create"))
    assert "-u" not in create_call and "--public" not in create_call  # must stay private
    push_call = next(c for c in calls if verb(c) == ("kernels", "push"))
    assert "--accelerator" in push_call

    run_slug = request["run_id"].lower()
    dataset_id = f"ocrkit-operator/ocrkit-rec-data-{run_slug}"
    kernel_id = f"ocrkit-operator/ocrkit-rec-{run_slug}"

    package_dir = Path(push_call[push_call.index("-p") + 1])
    kernel_metadata = json.loads((package_dir / "kernel-metadata.json").read_text())
    assert kernel_metadata["id"] == kernel_id
    assert kernel_metadata["is_private"] is True
    assert kernel_metadata["enable_gpu"] is True
    assert kernel_metadata["enable_internet"] is True
    assert kernel_metadata["dataset_sources"] == [dataset_id]

    dataset_package_dir = Path(create_call[create_call.index("-p") + 1])
    dataset_metadata = json.loads((dataset_package_dir / "dataset-metadata.json").read_text())
    assert dataset_metadata["id"] == dataset_id
    assert dataset_id != kernel_id  # Kaggle datasets/kernels share one per-account slug namespace

    run_dir = only_run(runs)
    assert (run_dir / "checkpoint/best_accuracy.pdparams").read_bytes() == b"weights"
    assert (run_dir / "checkpoint/config.yml").read_text() == "Global: {}\n"
    assert (run_dir / "remote.log").read_text() == "remote log contents\n"
    assert (run_dir / "evaluation/fixture_report.json").is_file()
    assert json.loads((run_dir / "run.json").read_text())["evaluation"]["field_accuracy"] == 0.99
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "success"
    assert status["input_dataset_deleted"] is True
    delete_call = next(c for c in calls if verb(c) == DATASETS_DELETE)
    assert delete_call[3] == dataset_metadata["id"]
    assert not (run_dir / "accepted").exists()
    assert fake_r2.deleted == [(request["checkpoint_upload"]["bucket"], request["checkpoint_upload"]["key"])]
    assert not fake_r2.objects


def test_kernel_failure_keeps_partial_checkpoint_and_deletes_dataset(kaggle_run) -> None:
    runs, calls, behavior, fake_r2 = kaggle_run
    behavior["kernel_status"] = "error"

    assert run_rec_kaggle.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert any(verb(c) == DATASETS_DELETE for c in calls)
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


def test_dataset_creation_failure_never_pushes_a_kernel_or_deletes_a_dataset(kaggle_run) -> None:
    runs, calls, behavior, fake_r2 = kaggle_run
    behavior["dataset_status"] = 1

    assert run_rec_kaggle.main() == 1

    assert not any(verb(c) == ("kernels", "push") for c in calls)
    assert not any(verb(c) == DATASETS_DELETE for c in calls)
    assert not (only_run(runs) / "checkpoint").exists()
    assert not fake_r2.objects


def test_dataset_cleanup_failure_is_reported_with_the_manual_command(kaggle_run) -> None:
    runs, _calls, behavior, _fake_r2 = kaggle_run
    behavior["delete_status"] = 1

    assert run_rec_kaggle.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    status = json.loads((run_dir / "status.json").read_text())
    assert "kaggle datasets delete" in status["error"]
    assert status["input_dataset_deleted"] is False


def test_kernel_output_retrieval_failure_is_reported_clearly(kaggle_run) -> None:
    runs, _calls, behavior, _fake_r2 = kaggle_run
    behavior["output_status"] = 1

    assert run_rec_kaggle.main() == 1

    error = json.loads((only_run(runs) / "status.json").read_text())["error"]
    assert "kernel output" in error


def test_poll_kernel_status_raises_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_rec_kaggle, "STATUS_POLL_SECONDS", 0)
    monkeypatch.setattr(run_rec_kaggle, "capture_command", lambda _c: SimpleNamespace(returncode=0, stdout="queued\n", stderr=""))

    with pytest.raises(RuntimeError, match="did not settle"):
        run_rec_kaggle.poll_kernel_status("kaggle", "owner/slug", 0.01, tmp_path / "log")


def test_username_resolved_from_kaggle_json_when_env_is_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "from-config", "key": "x"}), encoding="utf-8")
    parser = run_rec_kaggle.argparse.ArgumentParser()

    assert run_rec_kaggle.resolve_kaggle_username(parser) == "from-config"


def test_username_resolved_from_oauth_credentials_json_when_no_kaggle_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`kaggle auth login` (OAuth) stores credentials.json instead of the legacy kaggle.json."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "credentials.json").write_text(
        json.dumps({"username": "from-oauth", "access_token": "x", "refresh_token": "y"}), encoding="utf-8"
    )
    parser = run_rec_kaggle.argparse.ArgumentParser()

    assert run_rec_kaggle.resolve_kaggle_username(parser) == "from-oauth"


def test_remote_locate_input_archive_requires_exactly_one_attached_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training import kaggle_remote

    monkeypatch.setattr(kaggle_remote, "INPUT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="was not attached"):
        kaggle_remote.locate_input_archive()

    (tmp_path / "dataset-one").mkdir()
    (tmp_path / "dataset-one" / "ocrkit-input.tar.gz").write_bytes(b"a")
    assert kaggle_remote.locate_input_archive() == tmp_path / "dataset-one" / "ocrkit-input.tar.gz"

    (tmp_path / "dataset-two").mkdir()
    (tmp_path / "dataset-two" / "ocrkit-input.tar.gz").write_bytes(b"b")
    with pytest.raises(RuntimeError, match="exactly one"):
        kaggle_remote.locate_input_archive()


def test_remote_upload_checkpoint_raises_on_transport_failure(tmp_path: Path) -> None:
    from training import kaggle_remote

    checkpoint = tmp_path / "best_accuracy.pdparams"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="uploading the checkpoint"):
        kaggle_remote.upload_checkpoint(
            {"bucket": "b", "key": "k", "url": "http://127.0.0.1:1/unreachable"}, checkpoint
        )
