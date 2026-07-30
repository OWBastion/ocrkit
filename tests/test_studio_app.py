from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from training.studio import app as studio_app
from training.studio.app import create_app


def test_studio_api_serves_local_frontend_and_rejects_unknown_batch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend))

    assert client.get("/api/health").json() == {"ok": "true", "service": "ocrkit-studio"}
    assert client.get("/api/batches").json() == []
    assert client.get("/api/batches/not-a-batch/review").status_code == 404
    assert client.get("/").text == "<main>Studio</main>"


def test_studio_api_imports_image_into_private_batch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    image = cv2.imencode(".png", np.full((40, 60, 3), 200, dtype=np.uint8))[1].tobytes()
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.post("/api/batches", data={"holdout_ratio": "0.2"}, files=[("files", ("screenshot.png", image, "image/png"))])

    assert response.status_code == 200
    assert response.json()["batch"]["sources"] == 1
    assert len(list((tmp_path / "work/batches").iterdir())) == 1


def test_studio_api_adds_screenshot_to_existing_batch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    first = cv2.imencode(".png", np.full((40, 60, 3), 100, dtype=np.uint8))[1].tobytes()
    second = cv2.imencode(".png", np.full((40, 60, 3), 200, dtype=np.uint8))[1].tobytes()
    client = TestClient(create_app(tmp_path / "work", frontend))
    created = client.post("/api/batches", data={"holdout_ratio": "0.2"}, files=[("files", ("first.png", first, "image/png"))])
    batch_id = created.json()["batch"]["batch_id"]

    response = client.post(f"/api/batches/{batch_id}/sources", files=[("files", ("second.png", second, "image/png"))])

    assert response.status_code == 200
    assert response.json()["added"] == 1
    assert response.json()["batch"]["sources"] == 2


def test_studio_api_reports_missing_holdout_as_actionable_validation_error(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    batch = tmp_path / "work/batches/one-source"
    review = batch / "dataset/review"
    review.mkdir(parents=True)
    (batch / "batch.json").write_text(
        json.dumps({"sources": [{"id": "source-a", "split": "train"}]}), encoding="utf-8"
    )
    accepted = {"crop": "images/train/source-a/000.png", "review_status": "accepted", "transcription": "文字"}
    (review / "train.jsonl").write_text(json.dumps(accepted) + "\n", encoding="utf-8")
    (review / "holdout.jsonl").write_text("", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.post("/api/batches/one-source/finalize")

    assert response.status_code == 422
    assert "at least two distinct source screenshots" in response.json()["detail"]


def test_training_status_reaps_failed_child_process(monkeypatch) -> None:
    state: dict[str, object] = {"pid": 123, "status": "training"}
    monkeypatch.setattr(studio_app.os, "waitpid", lambda pid, options: (pid, 256))

    changed = studio_app._poll_training_process(state)

    assert changed is True
    assert state == {"pid": 123, "status": "failed", "exit_code": 1}


def test_studio_lists_only_complete_resume_checkpoints(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    batch = tmp_path / "work/batches/resume-batch"
    checkpoints = batch / "runs/smoke-20260730-104251/checkpoints"
    checkpoints.mkdir(parents=True)
    (batch / "batch.json").write_text("{}", encoding="utf-8")
    for suffix in (".pdparams", ".pdopt", ".states"):
        (checkpoints / f"best_accuracy{suffix}").write_text("checkpoint", encoding="utf-8")
    (checkpoints / "latest.pdparams").write_text("incomplete", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.get("/api/batches/resume-batch/training/checkpoints")

    assert response.status_code == 200
    assert response.json() == [{
        "path": "resume-batch:runs/smoke-20260730-104251/checkpoints/best_accuracy",
        "name": "resume-batch · smoke-20260730-104251/checkpoints/best_accuracy",
    }]


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
