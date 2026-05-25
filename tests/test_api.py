from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.context import AppContext, get_context
from app.main import app
from app.ocr.engine import OcrResult


class StubEngine:
    def recognize(self, image: np.ndarray) -> OcrResult:
        return OcrResult(text="", confidence=0.5, chunks=[])


def _dummy_png_bytes() -> bytes:
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000C49444154789C63606060000000040001F61738550000000049454E44AE426082"
    )


def _stub_context() -> AppContext:
    from app.core.roi_config import load_map_names, load_roi_config

    return AppContext(
        roi_config=load_roi_config(Path("configs/roi_1280x720.yaml")),
        map_names=load_map_names(Path("configs/maps.yaml")),
        ocr_engine=StubEngine(),
    )


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True


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
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["debug"] is not None
    assert "left_panel.deaths_skips_missing" in payload["warnings"]
    app.dependency_overrides.clear()
