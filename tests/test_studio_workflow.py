from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from training.jev.records import OPTION_NONE_CORRECT, OPTION_NOT_VALID, compute_input_digest, record_from_review_row
from training.jev.runner import DecisionOutput
from training.studio import app as studio_app
from training.studio import workflow
from training.studio.core import review_rows, save_jev_suggestion


def _batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batches/example"
    (batch / "dataset/review").mkdir(parents=True)
    (batch / "dataset/images").mkdir()
    (batch / "batch.json").write_text(json.dumps({"layout_version": "v1", "sources": []}))
    (batch / "dataset/images/pending.png").write_bytes(b"pending crop")
    (batch / "dataset/images/manual.png").write_bytes(b"reviewed crop")
    rows = [{"crop": "images/pending.png", "source_id": "source1", "roi": "map_panel", "candidate_text": "Samoa", "review_status": "pending"}, {"crop": "images/manual.png", "source_id": "source2", "roi": "map_panel", "candidate_text": "wrong", "review_status": "accepted", "transcription": "human truth", "review_method": "human"}]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text("")
    return batch


def test_suggestions_are_digest_bound_and_do_not_become_labels(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    row = review_rows(batch, "train")[0]
    record = record_from_review_row(batch.name, "train", "v1", hashlib.sha256(b"pending crop").hexdigest(), row)
    suggestion = {"input_digest": compute_input_digest(record), "action": "accept", "selected_option": "Samoa"}
    assert save_jev_suggestion(batch, "train", row["crop"], suggestion)
    current = review_rows(batch, "train")
    assert current[0]["review_status"] == "pending"
    assert "transcription" not in current[0]
    assert current[1]["transcription"] == "human truth"
    assert not (batch / "dataset/labels").exists()
    (batch / "dataset/images/pending.png").write_bytes(b"changed image")
    assert not save_jev_suggestion(batch, "train", row["crop"], suggestion)
    assert not save_jev_suggestion(batch, "train", "images/manual.png", suggestion)


@pytest.mark.parametrize(("option", "expected_action", "error"), [("Samoa", "accept", False), (OPTION_NOT_VALID, "reject", False), (OPTION_NONE_CORRECT, "manual", False), (None, "error", True)])
def test_jev_worker_only_suggests_pending_rows_and_preserves_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str | None, expected_action: str, error: bool) -> None:
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    calls = []
    closed = []
    class Runner:
        def __init__(self, *args: object) -> None:
            pass
        def decide(self, record, image: Path, tokens: int) -> DecisionOutput:
            calls.append(record)
            return DecisionOutput(record_id=record.record_id, input_digest=compute_input_digest(record), runner="test", status="error" if error else "ok", reason_code="worker_failure" if error else "ok", image_tokens=tokens, options=record.options, selected_option=option)
        def close(self) -> None:
            closed.append(True)
    monkeypatch.setattr(workflow, "OmniMlxRunner", Runner)
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True, "model_dir": "local", "worker_python": "python"})
    assert workflow.run_jev(batch, run, 70) == int(error)
    assert len(calls) == 1
    assert calls[0].truth_status is None
    assert closed == [True]
    rows = review_rows(batch, "train")
    assert rows[0]["jev_suggestion"]["action"] == expected_action
    assert rows[0]["review_status"] == "pending"
    assert rows[1]["transcription"] == "human truth"
    assert json.loads((run / "result.json").read_text())["errors"] == int(error)


def test_worker_cannot_overwrite_a_human_decision_made_during_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    class Runner:
        def __init__(self, *args: object) -> None:
            pass
        def decide(self, record, image: Path, tokens: int) -> DecisionOutput:
            from training.studio.core import update_review_row
            update_review_row(batch, "train", record.crop, "accepted", "human correction")
            return DecisionOutput(record_id=record.record_id, input_digest=compute_input_digest(record), runner="test", status="ok", reason_code="ok", image_tokens=tokens, options=record.options, selected_option="Samoa")
        def close(self) -> None:
            pass
    monkeypatch.setattr(workflow, "OmniMlxRunner", Runner)
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True, "model_dir": "local", "worker_python": "python"})
    workflow.run_jev(batch, run, 70)
    row = review_rows(batch, "train")[0]
    assert row["transcription"] == "human correction"
    assert "jev_suggestion" not in row
    assert json.loads((run / "result.json").read_text())["saved"] == 0


def test_api_starts_jev_in_background_and_blocks_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    launched = []
    monkeypatch.setattr(studio_app, "jev_config", lambda: {"configured": True})
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda command, **kwargs: launched.append(command) or SimpleNamespace(pid=444))
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (0, 0))
    client = TestClient(studio_app.create_app(tmp_path, None))
    path = "/api/batches/example/review/jev"
    assert client.post(path, json={"image_tokens": 19}).status_code == 422
    response = client.post(path, json={"image_tokens": 70})
    assert response.status_code == 200
    assert response.json()["status"] == "reviewing"
    assert launched[0][2:5] == ["-m", "training.studio.workflow", "jev"]
    assert client.post(path).status_code == 409
    assert len(launched) == 1
    state = response.json()
    Path(state["log"]).write_text("model loading")
    assert client.get(path).json()["log_tail"] == "model loading"
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (pid, 256))
    assert client.get(path).json()["status"] == "failed"
    assert (batch / "jev/latest.json").is_file()


def test_kaggle_api_uses_finalized_dataset_and_surfaces_live_remote_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    finalized = []
    launched = []
    monkeypatch.setattr(studio_app, "finalize_dataset", lambda directory: finalized.append(directory))
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda command, **kwargs: launched.append(command) or SimpleNamespace(pid=444))
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (0, 0))
    client = TestClient(studio_app.create_app(tmp_path, None))
    path = "/api/batches/example/training/kaggle"
    response = client.post(path, json={"epochs": 3})
    assert response.status_code == 200
    state = response.json()
    assert finalized == [batch.resolve()]
    assert state["backend"] == "kaggle"
    assert launched[0][-2:] == ["--epochs", "3"]
    assert client.post(path).status_code == 409
    assert client.post("/api/batches/example/training/smoke").status_code == 409
    output = Path(state["log"]).parent / "output"
    output.mkdir(parents=True)
    (output / "kaggle.log").write_text("kernel is queued")
    current = client.get("/api/batches/example/training").json()
    assert current["output_run_dir"] == str(output)
    assert "kernel is queued" in current["log_tail"]
    checkpoint = output / "checkpoint/best_accuracy"
    checkpoint.parent.mkdir()
    checkpoint.with_suffix(".pdparams").write_bytes(b"verified checkpoint")
    (Path(state["log"]).parent / "result.json").write_text(json.dumps({"checkpoint": str(checkpoint), "kaggle_kernel": "owner/kernel", "status": "success"}))
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (pid, 0))
    current = client.get("/api/batches/example/training").json()
    assert current["status"] == "completed"
    assert current["remote_status"] == "success"
    assert current["checkpoint"] == str(checkpoint)
    assert studio_app._checkpoint_from_training_state(batch) == checkpoint


def test_kaggle_rejects_unfinalizable_labels_without_starting_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _batch(tmp_path)
    def finalize(directory: Path) -> None:
        raise ValueError("human review still pending")
    monkeypatch.setattr(studio_app, "finalize_dataset", finalize)
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    client = TestClient(studio_app.create_app(tmp_path, None))
    response = client.post("/api/batches/example/training/kaggle", json={"epochs": 3})
    assert response.status_code == 422
    assert "pending" in response.json()["detail"]


def test_terminal_worker_result_survives_studio_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    run = batch / "runs/kaggle-run"
    run.mkdir(parents=True)
    state = {"pid": 444, "status": "training", "backend": "kaggle", "log": str(run / "training.log")}
    (batch / "runs/latest.json").write_text(json.dumps(state))
    workflow._write_result(run, {"workflow_status": "completed", "exit_code": 0, "status": "success"})
    def no_child(pid: int, flags: int):
        raise ChildProcessError()
    def no_process(pid: int, signal: int):
        raise ProcessLookupError()
    monkeypatch.setattr(studio_app.os, "waitpid", no_child)
    monkeypatch.setattr(studio_app.os, "kill", no_process)
    client = TestClient(studio_app.create_app(tmp_path, None))
    state = client.get("/api/batches/example/training").json()
    assert state["status"] == "completed"
    assert state["exit_code"] == 0


def test_candidate_replacement_and_teacher_adoption_blocked_during_jev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    (batch / "jev").mkdir()
    (batch / "jev/latest.json").write_text(json.dumps({"pid": 444, "status": "reviewing", "log": str(batch / "jev/log")}))
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (0, 0))
    client = TestClient(studio_app.create_app(tmp_path, None))
    for suffix in ("candidates", "candidates/recreate", "candidates/refresh-vision", "candidates/refresh-teacher", "review/accept-teacher"):
        assert client.post(f"/api/batches/example/{suffix}").status_code == 409
    response = client.put("/api/batches/example/review", json={"split": "train", "crop": "images/pending.png", "status": "accepted", "transcription": "human truth"})
    assert response.status_code == 200
    assert review_rows(batch, "train")[0]["transcription"] == "human truth"


def test_kaggle_wrapper_calls_existing_runner_with_exact_output_and_records_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        output = Path(command[-1])
        (output / "checkpoint").mkdir(parents=True)
        (output / "status.json").write_text(json.dumps({"status": "success", "kaggle_kernel": "owner/kernel"}))
        (output / "checkpoint/best_accuracy.pdparams").write_bytes(b"downloaded checkpoint")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess, "run", execute)
    assert workflow.run_kaggle(batch, run, 3) == 0
    assert "--labels-dir" in calls[0]
    assert calls[0][-4:] == ["--epochs", "3", "--output-dir", str(run / "output")]
    result = json.loads((run / "result.json").read_text())
    assert result["checkpoint"] == str(run / "output/checkpoint/best_accuracy")
    assert result["kaggle_kernel"] == "owner/kernel"


def test_worker_persists_failure_on_model_or_runner_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(workflow.sys, "argv", ["workflow", "jev", "--batch-dir", str(tmp_path), "--run-dir", str(run)])
    def failed(*args):
        raise ValueError("model not available")
    monkeypatch.setattr(workflow, "run_jev", failed)
    with pytest.raises(ValueError):
        workflow.main()
    assert json.loads((run / "result.json").read_text()) == {"workflow_status": "failed", "exit_code": 1}


def test_review_file_lock_serializes_a_separate_human_review_process(tmp_path: Path) -> None:
    import subprocess
    import sys
    from training.studio.core import _review_lock

    batch = _batch(tmp_path)
    code = "from pathlib import Path; from training.studio.core import update_review_row; import sys; print('ready',flush=True); update_review_row(Path(sys.argv[1]),'train','images/pending.png','accepted','human correction')"
    process = None
    try:
        with _review_lock(batch, "train"):
            process = subprocess.Popen([sys.executable, "-c", code, str(batch)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
            assert review_rows(batch, "train")[0]["review_status"] == "pending"
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert review_rows(batch, "train")[0]["transcription"] == "human correction"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_jev_configuration_preserves_virtualenv_interpreter_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    base = tmp_path / "base-python"
    base.write_text("base interpreter")
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(base)
    monkeypatch.setenv("OCRKIT_JEV_MODEL_DIR", str(model))
    monkeypatch.setenv("OCRKIT_JEV_WORKER_PYTHON", str(interpreter))
    config = workflow.jev_config()
    assert config["configured"] is True
    assert config["worker_python"] == str(interpreter)
    assert config["worker_python"] != str(base)


@pytest.mark.parametrize("engine", ["vision", "teacher"])
def test_refreshing_engine_inputs_discards_saved_jev_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    from training.studio import core
    batch = _batch(tmp_path)
    row = review_rows(batch, "train")[0]
    record = record_from_review_row(batch.name, "train", "v1", hashlib.sha256(b"pending crop").hexdigest(), row)
    assert save_jev_suggestion(batch, "train", row["crop"], {"input_digest": compute_input_digest(record), "selected_option": "Samoa", "action": "accept"})
    monkeypatch.setattr(core, "decode_image", lambda data: None)
    class Vision:
        def recognize(self, image):
            return [SimpleNamespace(text="a new engine option", confidence=0.8)]
    class Teacher:
        def __call__(self, image, **kwargs):
            return SimpleNamespace(txts=("a new engine option",), scores=(0.8,))
    if engine == "vision":
        core.refresh_vision_candidates(batch, vision_factory=Vision)
    else:
        core.refresh_teacher_candidates(batch, teacher_factory=Teacher)
    rows = review_rows(batch, "train")
    assert "jev_suggestion" not in rows[0]
    assert rows[0][f"{engine}_text"] == "a new engine option"
    assert rows[1]["transcription"] == "human truth"


def test_feedback_route_rejects_invalid_cutoff_before_reader_and_returns_optional_marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from training.studio import feedback
    calls = []
    monkeypatch.setattr(feedback, "get_feedback", lambda since: calls.append(since) or {"available": True, "marks": {"uploads/example.png": "inaccurate"}})
    client = TestClient(studio_app.create_app(tmp_path, None))
    assert client.get("/api/r2/feedback?since=invalid").status_code == 422
    assert calls == []
    response = client.get("/api/r2/feedback?since=2026-09-03")
    assert response.status_code == 200
    assert response.json() == {"available": True, "marks": {"uploads/example.png": "inaccurate"}}
    assert calls == ["2026-09-03"]


def test_selected_accuracy_marks_follow_r2_provenance_without_becoming_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    captured = []
    class Store:
        def download_images(self, keys, destination):
            image = destination / "input.png"
            image.write_bytes(b"source screenshot")
            return [SimpleNamespace(path=image, provenance={"sha256": "a" * 64, "object_key": "uploads/selected.png"})]
    def create(paths, **kwargs):
        captured.append(kwargs["provenance_by_digest"])
        return batch, {"id": "example"}
    monkeypatch.setattr(studio_app, "create_batch", create)
    client = TestClient(studio_app.create_app(tmp_path, None, remote_store=Store()))
    response = client.post("/api/batches/r2", json={"keys": ["uploads/selected.png"], "accuracy_marks": {"uploads/selected.png": "inaccurate", "uploads/unselected.png": "accurate"}})
    assert response.status_code == 200
    assert captured == [{"a" * 64: {"sha256": "a" * 64, "object_key": "uploads/selected.png", "accuracy": "inaccurate"}}]
    assert review_rows(batch, "train")[0]["review_status"] == "pending"
    assert "transcription" not in review_rows(batch, "train")[0]


def _suggest(batch: Path, split: str, crop: str, **suggestion: object) -> None:
    path = batch / "dataset/review" / f"{split}.jsonl"
    rows = review_rows(batch, split)
    next(row for row in rows if row["crop"] == crop)["jev_suggestion"] = {"status": "ok", **suggestion}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_confident_jev_choices_are_adopted_for_train_only(tmp_path: Path) -> None:
    from training.studio.core import adopt_jev_suggestions

    batch = _batch(tmp_path)
    rows = review_rows(batch, "train")
    rows += [
        {"crop": f"images/{name}.png", "source_id": name, "roi": "map_panel", "review_status": "pending"}
        for name in ("low", "reject")
    ]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text(json.dumps({"crop": "images/h.png", "source_id": "h", "roi": "map_panel", "review_status": "pending", "jev_suggestion": {"status": "ok", "action": "accept", "selected_option": "Samoa", "confidence": 0.99}}) + "\n")
    _suggest(batch, "train", "images/pending.png", action="accept", selected_option="Samoa", confidence=0.95)
    _suggest(batch, "train", "images/manual.png", action="accept", selected_option="jev guess", confidence=0.99)
    _suggest(batch, "train", "images/low.png", action="accept", selected_option="maybe", confidence=0.5)
    _suggest(batch, "train", "images/reject.png", action="reject", selected_option="not valid", confidence=0.97)
    _suggest(batch, "train", "images/manual.png", action="accept", selected_option="jev guess", confidence=0.99)
    _suggest(batch, "train", "images/manual.png", action="manual", selected_option=None, confidence=0.99)

    assert adopt_jev_suggestions(batch) == {"accepted": 1, "rejected": 1}
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/pending.png"]["review_status"] == "accepted"
    assert by_crop["images/pending.png"]["transcription"] == "Samoa"
    assert by_crop["images/pending.png"]["review_method"] == "automatic"
    assert by_crop["images/reject.png"]["review_status"] == "rejected"
    assert by_crop["images/low.png"]["review_status"] == "pending"
    assert by_crop["images/manual.png"]["transcription"] == "human truth"
    assert review_rows(batch, "holdout")[0]["review_status"] == "pending"


def test_prepare_imports_only_new_recent_screenshots_and_adopts_jev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = tmp_path / "batches/old"
    old.mkdir(parents=True)
    (old / "batch.json").write_text(json.dumps({"sources": [{"provenance": {"object_key": "uploads/submissions/seen.png"}}]}))
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    created = {}

    class Store:
        max_objects = 200
        def list_recent(self, since: str):
            return [{"key": "uploads/submissions/new.png"}, {"key": "uploads/submissions/seen.png"}]
        def download_images(self, keys, directory, progress_callback=None):
            created["keys"] = keys
            path = directory / "0001.png"
            path.write_bytes(b"png")
            progress_callback(1, 1, keys[0])
            return [SimpleNamespace(path=path, provenance={"object_key": keys[0], "sha256": "digest"})]

    def create(paths, **kwargs):
        created["provenance"] = kwargs["provenance_by_digest"]
        return batch, {}

    def jev(batch_dir, run_dir, tokens, on_progress):
        _suggest(batch_dir, "train", "images/pending.png", action="accept", selected_option="Samoa", confidence=0.95)
        on_progress(1, 1, 0)
        return 0

    monkeypatch.setattr(workflow.StudioR2Store, "from_settings", classmethod(lambda cls: Store()))
    monkeypatch.setattr(workflow, "get_feedback", lambda since: {"available": True, "marks": {"uploads/submissions/new.png": "inaccurate"}})
    monkeypatch.setattr(workflow, "create_batch", create)
    monkeypatch.setattr(workflow, "generate_candidates", lambda batch_dir: {})
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True})
    monkeypatch.setattr(workflow, "run_jev", jev)

    assert workflow.run_prepare(tmp_path, run, 30, 0.2, 70) == 0
    assert created["keys"] == ["uploads/submissions/new.png"]
    assert created["provenance"]["digest"]["accuracy"] == "inaccurate"
    result = json.loads((run / "result.json").read_text())
    assert result["stage"] == "done"
    assert result["batch_id"] == "example"
    assert result["jev_adopted"] == {"accepted": 1, "rejected": 0}
    assert review_rows(batch, "train")[0]["review_status"] == "accepted"


def test_prepare_with_nothing_new_finishes_on_latest_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _batch(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    class Store:
        max_objects = 200
        def list_recent(self, since: str):
            return []
    monkeypatch.setattr(workflow.StudioR2Store, "from_settings", classmethod(lambda cls: Store()))
    assert workflow.run_prepare(tmp_path, run, 30, 0.2, 70) == 0
    assert json.loads((run / "result.json").read_text()) == {"stage": "done", "new_screenshots": 0, "batch_id": batch.name}


def test_pipeline_api_previews_new_screenshots_and_blocks_duplicate_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _batch(tmp_path)
    launched = []
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda command, **kwargs: launched.append(command) or SimpleNamespace(pid=555))
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, flags: (0, 0))

    class Store:
        def list_recent(self, since: str):
            return [{"key": "a.png"}, {"key": "b.png"}]

    client = TestClient(studio_app.create_app(tmp_path, None, remote_store=Store()))
    assert client.get("/api/pipeline/preview").json() == {"new": 2, "recent": 2, "latest_batch": "example"}
    started = client.post("/api/pipeline/prepare", json={"days": 7})
    assert started.status_code == 200
    assert launched[0][2:5] == ["-m", "training.studio.workflow", "prepare"] and launched[0][launched[0].index("--days") + 1] == "7"
    assert client.post("/api/pipeline/prepare").status_code == 409
    assert client.get("/api/pipeline/status").json()["status"] == "preparing"


def test_engine_agreement_skips_jev_and_is_adopted_for_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from training.studio.core import adopt_engine_agreement, engines_agree

    batch = _batch(tmp_path)
    agree = {"rapidocr_text": "Samoa ", "vision_text": "Samoa", "review_status": "pending", "roi": "map_panel"}
    rows = [
        {**agree, "crop": "images/agree.png", "source_id": "a"},
        {**agree, "crop": "images/teacher.png", "source_id": "b", "teacher_text": "Other"},
        {**agree, "crop": "images/differ.png", "source_id": "c", "vision_text": "Samba"},
        {**agree, "crop": "images/single.png", "source_id": "d", "vision_text": None},
    ]
    holdout = [{**agree, "crop": "images/h.png", "source_id": "h"}]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (batch / "dataset/review/holdout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in holdout))
    assert [engines_agree(row) for row in rows] == [True, False, False, False]

    seen = []
    class Runner:
        def __init__(self, *args: object) -> None:
            pass
        def decide(self, record, image: Path, tokens: int) -> DecisionOutput:
            seen.append(record.record_id)
            return DecisionOutput(record_id=record.record_id, input_digest=compute_input_digest(record), runner="test", status="ok", reason_code="ok", image_tokens=tokens, options=record.options, selected_option=OPTION_NONE_CORRECT)
        def close(self) -> None:
            pass
    for name in ("agree", "teacher", "differ", "single", "h"):
        (batch / f"dataset/images/{name}.png").write_bytes(name.encode())
    monkeypatch.setattr(workflow, "OmniMlxRunner", Runner)
    monkeypatch.setattr(workflow, "jev_config", lambda: {"configured": True, "model_dir": "local", "worker_python": "python"})
    run = tmp_path / "run"
    run.mkdir()
    workflow.run_jev(batch, run, 70)
    assert len(seen) == 3  # teacher/differ/single only; agreeing Train and Holdout rows skipped

    assert adopt_engine_agreement(batch) == 1
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/agree.png"]["review_status"] == "accepted"
    assert by_crop["images/agree.png"]["transcription"] == "Samoa"
    assert by_crop["images/teacher.png"]["review_status"] == "pending"
    assert review_rows(batch, "holdout")[0]["review_status"] == "pending"


def test_studio_does_not_collect_center_banner_but_production_layout_keeps_it(tmp_path: Path) -> None:
    from app.core.roi_config import load_roi_config
    from training.studio.core import DEFAULT_ROI_CONFIG, _load_studio_roi_config, drop_excluded_rois

    assert "center_banner" in load_roi_config(DEFAULT_ROI_CONFIG).rois
    assert "center_banner" not in _load_studio_roi_config(DEFAULT_ROI_CONFIG).rois

    batch = _batch(tmp_path)
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"crop": "images/a.png", "roi": "center_banner", "review_status": "accepted", "transcription": "x"},
        {"crop": "images/b.png", "roi": "left_panel", "review_status": "pending"},
    ]))
    assert drop_excluded_rois(batch) == 1
    assert [row["roi"] for row in review_rows(batch, "train")] == ["left_panel"]
    assert (batch / "dataset/labels/train.txt").read_text() == ""


def test_ai_watermark_rejects_the_rest_of_that_screenshot_but_keeps_the_mark_and_human_work(tmp_path: Path) -> None:
    from training.studio.core import reject_ai_generated_sources, update_review_row

    batch = _batch(tmp_path)
    (batch / "batch.json").write_text(json.dumps({"layout_version": "v1", "sources": [{"id": "ai"}, {"id": "real"}]}))
    rows = [
        {"crop": "images/mark.png", "source_id": "ai", "roi": "ai_watermark", "rapidocr_text": "豆包AI生成", "review_status": "pending"},
        {"crop": "images/left.png", "source_id": "ai", "roi": "left_panel", "review_status": "pending"},
        {"crop": "images/done.png", "source_id": "ai", "roi": "left_panel", "review_status": "accepted", "transcription": "kept", "review_method": "human"},
        {"crop": "images/real.png", "source_id": "real", "roi": "left_panel", "review_status": "pending"},
        {"crop": "images/lookalike.png", "source_id": "real", "roi": "ai_watermark", "rapidocr_text": "AI", "review_status": "pending"},
    ]
    (batch / "dataset/review/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert reject_ai_generated_sources(batch) == 1
    by_crop = {row["crop"]: row for row in review_rows(batch, "train")}
    assert by_crop["images/left.png"]["review_status"] == "rejected"
    assert by_crop["images/left.png"]["auto_reject_reason"] == "ai_generated"
    assert by_crop["images/mark.png"]["review_status"] == "pending"
    assert by_crop["images/done.png"]["transcription"] == "kept"
    assert by_crop["images/real.png"]["review_status"] == "pending"
    assert [s.get("ai_generated") for s in json.loads((batch / "batch.json").read_text())["sources"]] == [True, None]

    row = update_review_row(batch, "train", "images/left.png", "accepted", "x", "event")
    assert row["category"] == "event"
    with pytest.raises(ValueError, match="category"):
        update_review_row(batch, "train", "images/left.png", "accepted", "x", "nope")


def test_health_tells_which_checkout_is_serving(tmp_path: Path) -> None:
    payload = TestClient(studio_app.create_app(tmp_path, None)).get("/api/health").json()
    assert payload["service"] == "ocrkit-studio"
    assert payload["root"] == str(studio_app.ROOT)
    assert "@" in payload["build"] or payload["build"] == "unknown"
