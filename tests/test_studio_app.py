from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from training.studio import app as studio_app
from training.studio.app import create_app
from training.studio.r2 import StudioR2Store


class _FakeR2Client:
    def list_objects(self, bucket: str, prefix: str, continuation_token: str | None = None, max_keys: int = 100) -> dict[str, object]:
        assert bucket == "evidence"
        assert prefix == "uploads/"
        return {
            "Contents": [
                {"Key": "uploads/one.png", "Size": 100},
                {"Key": "uploads/submissions/submission-1/hash.upload", "Size": 100},
                {"Key": "uploads/notes.txt", "Size": 100},
            ],
            "IsTruncated": False,
        }

    def get_object_bytes(self, bucket: str, key: str, version_id: str | None = None, *, max_bytes: int | None = None) -> bytes:
        assert bucket == "evidence"
        assert key in {"uploads/one.png", "uploads/submissions/submission-1/hash.upload"}
        return cv2.imencode(".png", np.full((40, 60, 3), 200, dtype=np.uint8))[1].tobytes()


def _remote_store() -> StudioR2Store:
    return StudioR2Store(_FakeR2Client(), "evidence", ("uploads/",), 200, 25 * 1024 * 1024)


def test_studio_model_promotion_requires_explicit_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    monkeypatch.setenv("OCRKIT_R2_DEFAULT_BUCKET", "models")
    monkeypatch.setenv("OCRKIT_R2_ENDPOINT_URL", "https://example.invalid")
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.post("/api/model-release/promote", json={"confirmed": False})

    assert response.status_code == 422
    assert "确认候选证据" in response.json()["detail"]


def test_studio_api_rejects_r2_image_outside_allowlist(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    response = client.get("/api/r2/image", params={"key": "../private.png"})

    assert response.status_code == 503
    assert response.json()["detail"] == "R2 bucket 或 prefix 不在 Studio 白名单内"


def test_studio_api_rejects_r2_key_outside_allowlist(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    response = client.post("/api/batches/r2", json={"keys": ["../private.png"]})

    assert response.status_code == 503
    assert response.json()["detail"] == "R2 bucket 或 prefix 不在 Studio 白名单内"


def test_studio_starts_confirmed_publication_from_latest_passed_checkpoint(tmp_path: Path, monkeypatch) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    batch = tmp_path / "work/batches/publish-batch"
    checkpoint = batch / "runs/smoke-1/checkpoints/best_accuracy"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.with_suffix(".pdparams").write_text("checkpoint", encoding="utf-8")
    (batch / "batch.json").write_text("{}", encoding="utf-8")
    (batch / "runs/latest.json").write_text(json.dumps({"status": "completed", "exit_code": 0, "log": str(batch / "runs/smoke-1/training.log")}), encoding="utf-8")
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=456))
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.post("/api/batches/publish-batch/publication", json={"confirmed": True})

    assert response.status_code == 200
    assert response.json()["status"] == "publishing"
    assert response.json()["command"][-1] == str(checkpoint)


def _kaggle_state(tmp_path: Path, *, evaluated: bool, merged: bool) -> tuple[Path, Path, Path]:
    frontend = tmp_path / "frontend"
    frontend.mkdir(exist_ok=True)
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    batch = tmp_path / "work/batches/publish-batch"
    output = batch / "runs/kaggle-1/output"
    checkpoint = output / "checkpoint/best_accuracy"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.with_suffix(".pdparams").write_text("checkpoint", encoding="utf-8")
    if evaluated:
        (output / "evaluation").mkdir()
        (output / "evaluation/comparison_report.json").write_text("{}", encoding="utf-8")
    dataset = batch / "runs/kaggle-1/dataset" if merged else batch / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    if merged:
        (dataset / "merge_manifest.json").write_text("{}", encoding="utf-8")
    (batch / "batch.json").write_text("{}", encoding="utf-8")
    (batch / "runs/latest.json").write_text(json.dumps({
        "status": "completed", "exit_code": 0, "backend": "kaggle", "checkpoint": str(checkpoint),
        "log": str(batch / "runs/kaggle-1/training.log"), "dataset_dir": str(dataset),
    }), encoding="utf-8")
    return frontend, checkpoint, dataset


def test_publication_reuses_the_evaluation_that_compared_the_checkpoint_with_the_model_in_service(tmp_path: Path, monkeypatch) -> None:
    frontend, checkpoint, dataset = _kaggle_state(tmp_path, evaluated=True, merged=True)
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=457))
    client = TestClient(create_app(tmp_path / "work", frontend))

    command = client.post("/api/batches/publish-batch/publication", json={"confirmed": True}).json()["command"]

    assert command[command.index("--evaluated-dir") + 1] == str(checkpoint.parents[1] / "evaluation")
    assert command[command.index("--provenance") + 1] == str(dataset / "merge_manifest.json")
    assert "--checkpoint" not in command


def test_publication_without_a_recorded_comparison_evaluates_inside_the_release(tmp_path: Path, monkeypatch) -> None:
    frontend, checkpoint, dataset = _kaggle_state(tmp_path, evaluated=False, merged=False)
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=458))
    client = TestClient(create_app(tmp_path / "work", frontend))

    command = client.post("/api/batches/publish-batch/publication", json={"confirmed": True}).json()["command"]

    assert "--evaluated-dir" not in command
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--holdout-labels") + 1] == str(dataset / "labels/holdout.txt")
