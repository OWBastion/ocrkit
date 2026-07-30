from __future__ import annotations

import json

import pytest

from app.core.context import AppContext
from app import main
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


def test_release_channel_refresh_swaps_only_after_new_model_loads(monkeypatch) -> None:
    context = AppContext(
        roi_config=None,  # type: ignore[arg-type]
        map_names=[],
        map_aliases={},
        ocr_engine="old",  # type: ignore[arg-type]
        model_version="v1",
        model_manifest_key="models/pp-ocrv6-small/v1/manifest.json",
    )
    monkeypatch.setattr(main.settings, "model_release_channel_key", "models/pp-ocrv6-small/channels/stable.json")
    monkeypatch.setattr(main, "_model_store", lambda: object())
    monkeypatch.setattr(main, "_selected_model_manifest", lambda store: "models/pp-ocrv6-small/v2/manifest.json")
    monkeypatch.setattr(main, "_load_model", lambda manifest_key: ("v2", "config-v2"))
    monkeypatch.setattr(main, "_create_ocr_engine", lambda config_path: "new")

    refreshed = main._refresh_channel_model(context)

    assert refreshed is not None
    assert refreshed.model_version == "v2"
    assert refreshed.model_manifest_key.endswith("v2/manifest.json")
    assert refreshed.ocr_engine == "new"
