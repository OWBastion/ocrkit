from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.context import AppContext, get_context
from app.main import app
from app.ocr.engine import OcrResult
from app.storage.r2_client import (
    ObjectAccessDeniedError,
    ObjectDownloadError,
    ObjectNotFoundError,
    ObjectTimeoutError,
)


class StubEngine:
    def recognize(self, image: np.ndarray) -> OcrResult:
        return OcrResult(text="", confidence=0.5, chunks=[])


class StubObjectStore:
    def __init__(self, payload: bytes | None = None, err: Exception | None = None) -> None:
        self.payload = payload
        self.err = err
        self.last_bucket: str | None = None

    def resolve_bucket(self, bucket: str | None) -> str:
        self.last_bucket = bucket
        return bucket or "ocr-bucket"

    def get_object_bytes(self, bucket: str, object_key: str, version_id: str | None = None) -> bytes:
        if self.err is not None:
            raise self.err
        return self.payload or b""


def _dummy_png_bytes() -> bytes:
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000C49444154789C63606060000000040001F61738550000000049454E44AE426082"
    )


def _make_context(object_store: StubObjectStore | None = None) -> AppContext:
    from app.core.roi_config import load_map_aliases, load_map_names, load_roi_config

    return AppContext(
        roi_config=load_roi_config(Path("configs/roi_1280x720.yaml")),
        map_names=load_map_names(Path("configs/maps.yaml")),
        map_aliases=load_map_aliases(Path("configs/maps.yaml")),
        ocr_engine=StubEngine(),
        object_store=object_store,
    )

def _stub_context() -> AppContext:
    return _make_context()


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["engine"] == "rapidocr"
    assert res.json()["model_version"] == "builtin"
    assert res.json()["application_version"] == "0.1.0"
    assert res.json()["version"] == "0.1.0"


def test_invalid_type() -> None:
    app.dependency_overrides[get_context] = _stub_context
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge", files={"file": ("a.txt", b"x", "text/plain")})
    assert res.status_code == 400
    app.dependency_overrides.clear()


def test_extract_ok_with_debug() -> None:
    app.dependency_overrides[get_context] = _stub_context
    client = TestClient(app)
    res = client.post(
        "/api/v1/ocr/challenge?debug=true",
        files={"file": ("img.png", _dummy_png_bytes(), "image/png")},
        headers={"X-Request-ID": "request-upload-1"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["schema_version"] == "1"
    assert payload["request_id"] == "request-upload-1"
    assert payload["engine"] == "rapidocr"
    assert payload["model_version"] == "builtin"
    assert payload["layout_version"] == "1280x720-v1"
    assert payload["quality"] == {
        "original_size": [1, 1],
        "aspect_ratio": 1.0,
        "layout_confidence": 0.0,
        "cropped": True,
        "blur_score": 1.0,
        "normalized_size": [1280, 720],
        "layout_version": "1280x720-v1",
        "warnings": payload["warnings"],
    }
    assert payload["fields"]["player"]["status"] == "missing"
    assert payload["fields"]["player"]["confidence"] == 0.0
    assert payload["fields"]["player"]["source_roi"] == ["bottom_left_hero", "center_banner"]
    assert payload["debug"] is not None
    assert "left_panel.deaths_skips_missing" in payload["warnings"]
    app.dependency_overrides.clear()


def test_by_object_ok_with_debug() -> None:
    object_store = StubObjectStore(payload=_dummy_png_bytes())
    app.dependency_overrides[get_context] = lambda: _make_context(object_store)
    client = TestClient(app)
    res = client.post(
        "/api/v1/ocr/challenge/by-object",
        json={"object_key": "uploads/a.png", "bucket": "owbastion-codes-evidence", "debug": True},
        headers={"X-Request-ID": "request-object-1"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["request_id"] == "request-object-1"
    assert payload["schema_version"] == "1"
    assert payload["engine"] == "rapidocr"
    assert payload["model_version"] == "builtin"
    assert payload["layout_version"] == "1280x720-v1"
    assert set(payload["fields"]) == {
        "challenge_completed",
        "heroes_completed",
        "heroes_total",
        "player",
        "deaths",
        "skips",
        "duration_text",
        "duration_seconds",
        "map_name",
        "difficulty",
        "version",
    }
    assert payload["debug"] is not None
    assert object_store.last_bucket == "owbastion-codes-evidence"
    app.dependency_overrides.clear()


def test_by_object_invalid_key() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "../x.png"})
    assert res.status_code == 400
    app.dependency_overrides.clear()


def test_by_object_rejects_reserved_model_prefix() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "models/pp-ocrv6-small/v1/det.onnx"})

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "INVALID_OBJECT_KEY"
    app.dependency_overrides.clear()


def test_by_object_rejects_platform_legacy_prefix() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = TestClient(app)
    res = client.post(
        "/api/v1/ocr/challenge/by-object",
        json={"object_key": "evidence/submissions/submission-1/image.upload"},
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "INVALID_OBJECT_KEY"
    app.dependency_overrides.clear()


def test_by_object_store_unavailable() -> None:
    app.dependency_overrides[get_context] = _stub_context
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/x.png"})
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "OBJECT_STORE_UNAVAILABLE"
    app.dependency_overrides.clear()


def test_by_object_not_found() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(err=ObjectNotFoundError("missing")))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/x.png"})
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "OBJECT_NOT_FOUND"
    app.dependency_overrides.clear()


def test_by_object_access_denied() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(err=ObjectAccessDeniedError("denied")))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/x.png"})
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "OBJECT_ACCESS_DENIED"
    app.dependency_overrides.clear()


def test_by_object_timeout() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(err=ObjectTimeoutError("timeout")))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/x.png"})
    assert res.status_code == 504
    assert res.json()["detail"]["code"] == "OBJECT_DOWNLOAD_TIMEOUT"
    app.dependency_overrides.clear()


def test_by_object_download_failed() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(err=ObjectDownloadError("failed")))
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/x.png"})
    assert res.status_code == 502
    assert res.json()["detail"]["code"] == "OBJECT_DOWNLOAD_FAILED"
    app.dependency_overrides.clear()
