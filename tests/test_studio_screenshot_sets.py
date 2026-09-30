from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from training.importer import (
    ScreenshotSetAuthError,
    ScreenshotSetContractError,
    ScreenshotSetMetadata,
    ScreenshotSetNotFinalizedError,
    ScreenshotSetNotFoundError,
)
from training.studio.app import create_app
from training.studio.r2 import StudioR2Store


def _png(shade: int = 200) -> bytes:
    return cv2.imencode(".png", np.full((720, 1280, 3), shade, dtype=np.uint8))[1].tobytes()


def _member(source_id: str, data: bytes, *, accuracy: str | None = None) -> dict[str, object]:
    return {
        "source_id": source_id,
        "object_key": f"evidence/screens/{source_id}.png",
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime_type": "image/png",
        "size_bytes": len(data),
        "layout_version": "1280x720-v6",
        "accuracy": accuracy,
    }


def _set_payload(members: list[dict[str, object]], **overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "set_id": "set-abc",
        "version": 3,
        "finalized": True,
        "finalized_at": "2026-08-01T12:00:00Z",
        "members": members,
    }
    payload.update(overrides)
    return payload


class _FakeSetClient:
    def __init__(self, payload: dict[str, object] | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.requested_versions: list[int] = []

    def fetch_set(self, version: int) -> ScreenshotSetMetadata:
        self.requested_versions.append(version)
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return ScreenshotSetMetadata.model_validate(self.payload)


class _FakeR2Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_object_bytes(self, bucket: str, key: str, version_id: str | None = None, *, max_bytes: int | None = None) -> bytes:
        assert bucket == "evidence"
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]


def _store(objects: dict[str, bytes]) -> StudioR2Store:
    return StudioR2Store(_FakeR2Client(objects), "evidence", ("evidence/",), 200, 25 * 1024 * 1024)


def _frontend(tmp_path: Path) -> Path:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    return frontend


def test_screenshot_set_import_creates_reviewable_batch(tmp_path: Path) -> None:
    one, two = _png(200), _png(64)
    payload = _set_payload([_member("src-1", one, accuracy="accurate"), _member("src-2", two, accuracy="inaccurate")])
    client = TestClient(
        create_app(
            tmp_path / "work",
            _frontend(tmp_path),
            remote_store=_store({"evidence/screens/src-1.png": one, "evidence/screens/src-2.png": two}),
            screenshot_set_client=_FakeSetClient(payload),
        )
    )

    response = client.post("/api/screenshot-sets/import", json={"version": 3, "holdout_ratio": 0.0})

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["set_id"] == "set-abc"
    assert body["report"]["version"] == 3
    assert body["report"]["member_count"] == 2
    assert body["report"]["accuracy_counts"] == {"accurate": 1, "inaccurate": 1, "unset": 0}
    assert body["batch"]["sources"] == 2
    assert body["batch"]["screenshot_set"]["set_id"] == "set-abc"
    assert body["batch"]["screenshot_set"]["version"] == 3

    batch_dir = tmp_path / "work/batches" / body["batch"]["batch_id"]
    manifest = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert manifest["screenshot_set"]["set_id"] == "set-abc"
    assert manifest["screenshot_set"]["version"] == 3
    provenance_by_source = {row["provenance"]["source_id"]: row["provenance"] for row in manifest["sources"]}
    assert provenance_by_source["src-1"]["object_key"] == "evidence/screens/src-1.png"
    assert provenance_by_source["src-1"]["set_id"] == "set-abc"
    assert provenance_by_source["src-2"]["accuracy"] == "inaccurate"
    cases = json.loads((batch_dir / "cases.json").read_text(encoding="utf-8"))
    feedback = {row["id"]: row["accuracy_feedback"] for row in cases}
    source_id_by_case = {row["id"]: row["provenance"]["source_id"] for row in manifest["sources"]}
    src2_case = next(case_id for case_id, source_id in source_id_by_case.items() if source_id == "src-2")
    assert feedback[src2_case] == "inaccurate"

    batches = client.get("/api/batches").json()
    assert batches[0]["screenshot_set"]["set_id"] == "set-abc"


def test_screenshot_set_import_is_not_repeatable(tmp_path: Path) -> None:
    one = _png(200)
    payload = _set_payload([_member("src-1", one)])
    app_client = TestClient(
        create_app(
            tmp_path / "work",
            _frontend(tmp_path),
            remote_store=_store({"evidence/screens/src-1.png": one}),
            screenshot_set_client=_FakeSetClient(payload),
        )
    )

    first = app_client.post("/api/screenshot-sets/import", json={"version": 3})
    assert first.status_code == 200
    second = app_client.post("/api/screenshot-sets/import", json={"version": 3})
    assert second.status_code == 409
    assert first.json()["batch"]["batch_id"] in second.json()["detail"]


def test_screenshot_set_import_requires_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCRKIT_SCREENSHOT_SET_BASE_URL", raising=False)
    monkeypatch.delenv("OCRKIT_SCREENSHOT_SET_TOKEN", raising=False)
    client = TestClient(create_app(tmp_path / "work", _frontend(tmp_path)))

    response = client.post("/api/screenshot-sets/import", json={"version": 3})

    assert response.status_code == 503
    assert "OCRKIT_SCREENSHOT_SET" in response.json()["detail"]


def test_screenshot_set_import_requires_r2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(StudioR2Store, "from_settings", staticmethod(lambda: None))
    client = TestClient(
        create_app(tmp_path / "work", _frontend(tmp_path), screenshot_set_client=_FakeSetClient(_set_payload([_member("src-1", _png())])))
    )

    response = client.post("/api/screenshot-sets/import", json={"version": 3})

    assert response.status_code == 503
    assert "R2" in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ScreenshotSetAuthError("denied"), 503),
        (ScreenshotSetNotFoundError("missing"), 404),
        (ScreenshotSetNotFinalizedError("not finalized"), 409),
        (ScreenshotSetContractError("bad contract"), 502),
    ],
)
def test_screenshot_set_import_maps_platform_errors(tmp_path: Path, error: Exception, status: int) -> None:
    client = TestClient(
        create_app(tmp_path / "work", _frontend(tmp_path), remote_store=_store({}), screenshot_set_client=_FakeSetClient(error=error))
    )

    response = client.post("/api/screenshot-sets/import", json={"version": 3})

    assert response.status_code == status


def test_screenshot_set_member_integrity_failure_is_422(tmp_path: Path) -> None:
    payload = _set_payload([_member("src-1", _png(200))])
    objects = {"evidence/screens/src-1.png": _png(10)}  # bytes do not match declared sha256
    client = TestClient(
        create_app(tmp_path / "work", _frontend(tmp_path), remote_store=_store(objects), screenshot_set_client=_FakeSetClient(payload))
    )

    response = client.post("/api/screenshot-sets/import", json={"version": 3})

    assert response.status_code == 422
    assert "src-1" in response.json()["detail"]


def test_screenshot_set_import_uses_named_resumable_workspace(tmp_path: Path) -> None:
    one = _png(200)
    payload = _set_payload([_member("src-1", one)])
    store = _store({"evidence/screens/src-1.png": one})
    client = TestClient(
        create_app(tmp_path / "work", _frontend(tmp_path), remote_store=store, screenshot_set_client=_FakeSetClient(payload))
    )

    response = client.post("/api/screenshot-sets/import", json={"version": 3})

    assert response.status_code == 200
    workspace = tmp_path / "work/set-workspace/set-3/objects"
    assert workspace.is_dir()
    assert list(workspace.iterdir())
