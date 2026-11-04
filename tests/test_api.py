from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.context import AppContext, get_context
from app.main import app
from app.ocr.engine import OcrResult
from app.service import extract_structured


class StubEngine:
    def recognize(self, image: np.ndarray) -> OcrResult:
        return OcrResult(text="", confidence=0.5, chunks=[])


class LowConfidenceRunCodeEngine:
    def recognize(self, image: np.ndarray) -> OcrResult:
        if image.shape[:2] == (172, 660):
            return OcrResult(text="本局代码：4821-7354-1926", confidence=0.89, chunks=[])
        return OcrResult(text="", confidence=0.5, chunks=[])


class AmbiguousRunCodeEngine:
    def recognize(self, image: np.ndarray) -> OcrResult:
        if image.shape[:2] == (172, 660):
            return OcrResult(
                text="本局代码：4821-7354-1926 Run Code: 4821-7354-1927",
                confidence=0.97,
                chunks=[],
            )
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


@pytest.fixture(autouse=True)
def _configure_service_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", "test-service-token")
    monkeypatch.setattr(settings, "allow_debug", True)


def _client() -> TestClient:
    return TestClient(app, headers={"Authorization": "Bearer test-service-token"})


def test_extract_rejects_low_confidence_run_code() -> None:
    context = _make_context()
    response = extract_structured(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        context.roi_config,
        context.map_names,
        context.map_aliases,
        LowConfidenceRunCodeEngine(),
        False,
        "request-run-code-low-confidence-1",
        "rapidocr",
        "builtin",
        context.roi_config.version,
    )

    assert response.data.run_code is None
    assert response.fields["run_code"].confidence == 0.89
    assert response.fields["run_code"].status == "low_confidence"
    assert "run_code.low_confidence" in response.warnings


def test_extract_marks_conflicting_run_code_candidates_ambiguous() -> None:
    context = _make_context()
    response = extract_structured(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        context.roi_config,
        context.map_names,
        context.map_aliases,
        AmbiguousRunCodeEngine(),
        False,
        "request-run-code-ambiguous-1",
        "rapidocr",
        "builtin",
        context.roi_config.version,
    )

    assert response.data.run_code is None
    assert response.fields["run_code"].status == "ambiguous"
    assert "run_code.ambiguous" in response.warnings


def test_recognition_requires_service_token() -> None:
    client = TestClient(app)
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "uploads/a.png"})
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "UNAUTHORIZED"


def test_debug_is_disabled_when_production_setting_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allow_debug", False)
    app.dependency_overrides[get_context] = _stub_context
    res = _client().post(
        "/api/v1/ocr/challenge?debug=true",
        files={"file": ("img.png", _dummy_png_bytes(), "image/png")},
    )
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "DEBUG_DISABLED"
    app.dependency_overrides.clear()


def test_by_object_invalid_key() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = _client()
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "../x.png"})
    assert res.status_code == 400
    app.dependency_overrides.clear()


def test_by_object_rejects_reserved_model_prefix() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = _client()
    res = client.post("/api/v1/ocr/challenge/by-object", json={"object_key": "models/pp-ocrv6-small/v1/det.onnx"})

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "INVALID_OBJECT_KEY"
    app.dependency_overrides.clear()


def test_by_object_rejects_platform_legacy_prefix() -> None:
    app.dependency_overrides[get_context] = lambda: _make_context(StubObjectStore(payload=_dummy_png_bytes()))
    client = _client()
    res = client.post(
        "/api/v1/ocr/challenge/by-object",
        json={"object_key": "evidence/submissions/submission-1/image.upload"},
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "INVALID_OBJECT_KEY"
    app.dependency_overrides.clear()
