from __future__ import annotations

import json

import pytest

from app.model_artifacts.channel import load_release_channel


class StubObjectStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get_object_bytes(self, bucket: str, object_key: str) -> bytes:
        return self.payload


def test_load_release_channel_resolves_versioned_manifest() -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "model": "pp-ocrv6-small",
            "manifest_key": "models/pp-ocrv6-small/2026.07.31-01/manifest.json",
        }
    ).encode()

    channel = load_release_channel(
        StubObjectStore(payload),  # type: ignore[arg-type]
        "models",
        "models/pp-ocrv6-small/channels/stable.json",
    )

    assert channel.manifest_key == "models/pp-ocrv6-small/2026.07.31-01/manifest.json"


def test_load_release_channel_rejects_non_model_manifest() -> None:
    payload = json.dumps(
        {"schema_version": 1, "model": "pp-ocrv6-small", "manifest_key": "uploads/model/manifest.json"}
    ).encode()

    with pytest.raises(ValueError, match="manifest key"):
        load_release_channel(
            StubObjectStore(payload),  # type: ignore[arg-type]
            "models",
            "models/pp-ocrv6-small/channels/stable.json",
        )
