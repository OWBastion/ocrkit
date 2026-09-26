from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

from training import run_rec_colab

SESSION_STOP = "stop"


def build_result_archive(path: Path, *, status: str = "success") -> None:
    files = {
        "results/checkpoint/best_accuracy.pdparams": b"weights",
        "results/evaluation/fixture_report.json": b'{"field_accuracy": 0.99}',
        "results/remote.log": b"log",
    }
    outputs = [
        {
            "path": name.removeprefix("results/"),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in files.items()
        if not name.endswith("remote.log")
    ]
    files["results/run.json"] = json.dumps({"status": status, "outputs": outputs}).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            source = path.parent / "member"
            source.write_bytes(data)
            archive.add(source, arcname=name)


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
    behavior = {"exec_status": 0, "result_status": "success", "stop_status": 0}

    def fake_stream(command: list[str], _log: Path) -> int:
        calls.append(command)
        verb = command[1]
        if verb == "exec":
            return behavior["exec_status"]
        if verb == "download":
            remote, local = command[-2], Path(command[-1])
            if "remote_result" not in behavior:
                archive = local.parent / "remote-result.tar.gz"
                build_result_archive(archive, status=behavior["result_status"])
                behavior["remote_result"] = archive.read_bytes()
                archive.unlink()
            data = behavior["remote_result"]
            parts = [data[i : i + 2000] for i in range(0, len(data), 2000)]
            if remote.endswith("index.json"):
                local.write_text(
                    json.dumps(
                        {
                            "parts": [
                                {"name": f"ocrkit-result.part{i:04d}", "sha256": hashlib.sha256(part).hexdigest()}
                                for i, part in enumerate(parts)
                            ]
                        }
                    )
                )
            else:
                local.write_bytes(parts[int(remote.rsplit("part", 1)[1])])
            return 0
        if verb == SESSION_STOP:
            return behavior["stop_status"]
        return 0

    monkeypatch.setattr(run_rec_colab, "RUNS", runs)
    monkeypatch.setattr(run_rec_colab.shutil, "which", lambda _name: "colab")
    monkeypatch.setattr(run_rec_colab, "validate_labels", lambda _path: (1, ["a.png"]))
    monkeypatch.setattr(run_rec_colab, "PART_BYTES", 4)
    monkeypatch.setattr(run_rec_colab, "stage_inputs", lambda archive, *_a: archive.write_bytes(b"input-archive"))
    monkeypatch.setattr(run_rec_colab, "stream_colab", fake_stream)
    monkeypatch.setattr(
        sys, "argv", ["run_rec_colab.py", "--labels-dir", str(labels), "--pretrained-checkpoint", str(checkpoint)]
    )
    return runs, calls, behavior


def only_run(runs: Path) -> Path:
    (run_dir,) = runs.iterdir()
    return run_dir


def test_success_keeps_verified_artifacts_and_stops_runtime(colab_run) -> None:
    runs, calls, _ = colab_run

    assert run_rec_colab.main() == 0

    uploads = [command[-1] for command in calls if command[1] == "upload"]
    assert uploads == [f"/content/ocrkit-input.part{i:04d}" for i in range(4)]
    exec_command = next(command for command in calls if command[1] == "exec")
    assert float(exec_command[exec_command.index("--timeout") + 1]) == 6 * 3600
    run_dir = only_run(runs)
    assert (run_dir / "checkpoint/best_accuracy.pdparams").is_file()
    assert (run_dir / "evaluation/fixture_report.json").is_file()
    assert json.loads((run_dir / "status.json").read_text())["runtime_stopped"] is True
    assert calls[-1][1] == SESSION_STOP
    assert not (run_dir / "accepted").exists()


def test_training_failure_keeps_partial_output_and_stops_runtime(colab_run) -> None:
    runs, calls, behavior = colab_run
    behavior.update(exec_status=1, result_status="failed")

    assert run_rec_colab.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert not (run_dir / "run.json").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert calls[-1][1] == SESSION_STOP
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"


def test_teardown_failure_demotes_accepted_artifacts_to_partial(colab_run) -> None:
    runs, _, behavior = colab_run
    behavior["stop_status"] = 1

    assert run_rec_colab.main() == 1

    run_dir = only_run(runs)
    assert not (run_dir / "checkpoint").exists()
    assert (run_dir / "partial/checkpoint/best_accuracy.pdparams").is_file()
    assert "colab stop" in json.loads((run_dir / "status.json").read_text())["error"]


def test_provisioning_failure_stops_runtime_and_uploads_nothing(colab_run) -> None:
    runs, calls, _ = colab_run
    original = run_rec_colab.stream_colab
    run_rec_colab.stream_colab = lambda command, log: 1 if command[1] == "new" else original(command, log)
    try:
        assert run_rec_colab.main() == 1
    finally:
        run_rec_colab.stream_colab = original

    assert [command[1] for command in calls] == [SESSION_STOP]
    assert not (only_run(runs) / "checkpoint").exists()


def test_tampered_result_is_rejected(tmp_path: Path) -> None:
    remote = tmp_path / "retrieved"
    archive = tmp_path / "result.tar.gz"
    build_result_archive(archive)
    run_rec_colab.safe_extract_result(archive, remote)
    (remote / "results/checkpoint/best_accuracy.pdparams").write_bytes(b"tampered")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="checksum"):
        run_rec_colab.copy_remote_result(remote, run_dir, success=True)

    assert not (run_dir / "accepted").exists()


def test_remote_parts_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from training import colab_remote

    monkeypatch.setattr(colab_remote, "CONTENT", tmp_path)
    monkeypatch.setattr(colab_remote, "INPUT_ARCHIVE", tmp_path / "in.tar.gz")
    monkeypatch.setattr(colab_remote, "RESULT_ARCHIVE", tmp_path / "out.tar.gz")
    monkeypatch.setattr(colab_remote, "RESULT_INDEX", tmp_path / "out.index.json")
    monkeypatch.setattr(colab_remote, "PART_BYTES", 3)
    payload = b"0123456789"
    for index in range(2):
        (tmp_path / f"ocrkit-input.part{index:04d}").write_bytes(payload[index * 5 : index * 5 + 5])
    colab_remote.join_input_parts()
    assert (tmp_path / "in.tar.gz").read_bytes() == payload
    assert not list(tmp_path.glob("ocrkit-input.part*"))

    (tmp_path / "out.tar.gz").write_bytes(payload)
    colab_remote.split_result_archive()
    index = json.loads((tmp_path / "out.index.json").read_text())
    assert b"".join((tmp_path / part["name"]).read_bytes() for part in index["parts"]) == payload
    assert len(index["parts"]) == 4
