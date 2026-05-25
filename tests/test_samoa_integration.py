from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("OCRKIT_BASE_URL", "http://127.0.0.1:8001")


def _wait_for_health(timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0, trust_env=False) as client:
                r = client.get(f"{BASE_URL}/health")
            if r.status_code == 200 and r.json().get("ok") is True:
                return
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError("service did not become healthy")


def test_samoa_image_expected_data() -> None:
    _wait_for_health()

    image_path = Path("tests/test-_samoa.png")
    with image_path.open("rb") as f:
        files = {"file": ("test-_samoa.png", f, "image/png")}
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            res = client.post(f"{BASE_URL}/api/v1/ocr/challenge?debug=true", files=files)

    assert res.status_code == 200
    payload = res.json()

    if payload.get("ok") is not True:
        raise AssertionError(json.dumps(payload, ensure_ascii=False, indent=2))

    data = payload["data"]
    assert data["challenge_completed"] is True
    assert data["heroes_completed"] == 51
    assert data["heroes_total"] == 51
    assert data["deaths"] == 114
    assert data["skips"] == 0
    assert data["duration_seconds"] == 8438.0
    assert data["map_name"] == "萨摩亚"
    assert data["difficulty"] == "地狱"
    assert data["version"] == "26.0513.6"

    # player name is optional for this sample
    assert data["player"] is None or isinstance(data["player"], str)
