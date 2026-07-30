from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

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
