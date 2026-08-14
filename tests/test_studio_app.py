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


def test_studio_api_lists_and_imports_r2_images_into_a_new_batch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    listed = client.get("/api/r2/images", params={"prefix": "uploads/"})
    assert listed.status_code == 200
    assert listed.json()["objects"] == [
        {"key": "uploads/one.png", "size": 100, "etag": None, "last_modified": None},
        {"key": "uploads/submissions/submission-1/hash.upload", "size": 100, "etag": None, "last_modified": None},
    ]

    imported = client.post("/api/batches/r2", json={"keys": ["uploads/one.png"], "holdout_ratio": 0.2})

    assert imported.status_code == 200
    batch_id = imported.json()["batch"]["batch_id"]
    manifest = json.loads((tmp_path / "work/batches" / batch_id / "batch.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["provenance"] == {
        "source": "r2",
        "bucket": "evidence",
        "object_key": "uploads/one.png",
        "sha256": manifest["sources"][0]["sha256"],
    }


def test_studio_api_streams_r2_import_progress(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    response = client.post("/api/batches/r2/stream", json={"keys": ["uploads/one.png"], "holdout_ratio": 0.2})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    lines = [json.loads(line) for line in response.text.strip().split("\n") if line.strip()]
    stages = [item.get("stage") for item in lines if item.get("type") == "progress"]
    assert "downloading" in stages
    assert "creating" in stages
    done = next(item for item in lines if item.get("type") == "done")
    assert done["batch"]["sources"] == 1


def test_studio_api_streams_r2_append_progress(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    local = cv2.imencode(".png", np.full((40, 60, 3), 100, dtype=np.uint8))[1].tobytes()
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))
    created = client.post("/api/batches", data={"holdout_ratio": "0.2"}, files=[("files", ("local.png", local, "image/png"))])
    batch_id = created.json()["batch"]["batch_id"]

    response = client.post(f"/api/batches/{batch_id}/remote-sources/stream", json={"keys": ["uploads/one.png"]})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    lines = [json.loads(line) for line in response.text.strip().split("\n") if line.strip()]
    done = next(item for item in lines if item.get("type") == "done")
    assert done["added"] == 1


def test_studio_api_serves_r2_image_preview(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    response = client.get("/api/r2/image", params={"key": "uploads/one.png"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert len(response.content) > 0


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


def test_studio_api_appends_r2_images_to_an_existing_batch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    local = cv2.imencode(".png", np.full((40, 60, 3), 100, dtype=np.uint8))[1].tobytes()
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))
    created = client.post("/api/batches", data={"holdout_ratio": "0.2"}, files=[("files", ("local.png", local, "image/png"))])
    batch_id = created.json()["batch"]["batch_id"]

    response = client.post(f"/api/batches/{batch_id}/remote-sources", json={"keys": ["uploads/one.png"]})

    assert response.status_code == 200
    assert response.json()["added"] == 1
    manifest = json.loads((tmp_path / "work/batches" / batch_id / "batch.json").read_text(encoding="utf-8"))
    assert len(manifest["sources"]) == 2
    assert manifest["sources"][1]["provenance"]["object_key"] == "uploads/one.png"


def test_studio_api_imports_platform_upload_objects(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend, _remote_store()))

    response = client.post("/api/batches/r2", json={"keys": ["uploads/submissions/submission-1/hash.upload"]})

    assert response.status_code == 200
    batch_id = response.json()["batch"]["batch_id"]
    manifest = json.loads((tmp_path / "work/batches" / batch_id / "batch.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["file"].endswith(".png")


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


def test_studio_lists_complete_legacy_checkpoints(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    legacy = tmp_path / "checkpoints/rec_pp_ocrv6_small/best_accuracy"
    legacy.parent.mkdir(parents=True)
    for suffix in (".pdparams", ".pdopt", ".states"):
        (legacy.parent / f"best_accuracy{suffix}").write_text("checkpoint", encoding="utf-8")
    batch = tmp_path / "work/batches/current"
    batch.mkdir(parents=True)
    (batch / "batch.json").write_text("{}", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "work", frontend))

    response = client.get("/api/batches/current/training/checkpoints")

    assert response.status_code == 200
    assert response.json() == [{
        "path": "legacy:rec_pp_ocrv6_small/best_accuracy",
        "name": "历史模型 · rec_pp_ocrv6_small/best_accuracy",
    }]


def test_studio_resolves_legacy_checkpoint_only_inside_legacy_root(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    batch = work_root / "batches/current"
    batch.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoints/rec_pp_ocrv6_small/best_accuracy"
    checkpoint.parent.mkdir(parents=True)
    for suffix in (".pdparams", ".pdopt", ".states"):
        (checkpoint.parent / f"best_accuracy{suffix}").write_text("checkpoint", encoding="utf-8")

    assert studio_app._resume_checkpoint(work_root, batch, "legacy:rec_pp_ocrv6_small/best_accuracy") == checkpoint
    with pytest.raises(studio_app.HTTPException):
        studio_app._resume_checkpoint(work_root, batch, "legacy:../outside")


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
