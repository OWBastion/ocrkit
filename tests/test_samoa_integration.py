from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = os.getenv("OCRKIT_BASE_URL", "http://127.0.0.1:8001")
CASES_PATH = Path("datasets/fixtures/challenge/cases.json")
FIXTURE_DIR = CASES_PATH.parent


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


def _load_cases() -> list[dict]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise AssertionError("cases.json must be a non-empty array")
    return raw


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_challenge_cases(case: dict) -> None:
    _wait_for_health()

    image_name = case["image"]
    image_path = FIXTURE_DIR / image_name
    expected = case["expected"]

    with image_path.open("rb") as f:
        files = {"file": (image_name, f, "image/png")}
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            res = client.post(f"{BASE_URL}/api/v1/ocr/challenge?debug=true", files=files)

    assert res.status_code == 200, f"case={case['id']}, status={res.status_code}"
    payload = res.json()

    if payload.get("ok") is not True:
        raise AssertionError(f"case={case['id']}\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    data = payload["data"]
    for key, value in expected.items():
        assert data[key] == value, f"case={case['id']}, key={key}, got={data[key]}, expected={value}"

    assert data["player"] is None or isinstance(data["player"], str)
