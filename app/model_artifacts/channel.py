from __future__ import annotations

import json
from dataclasses import dataclass

from app.storage.r2_client import R2ObjectStore

from .constants import MODEL_OBJECT_PREFIX


@dataclass(frozen=True)
class ModelReleaseChannel:
    manifest_key: str


def load_release_channel(store: R2ObjectStore, bucket: str, channel_key: str) -> ModelReleaseChannel:
    expected_prefix = f"{MODEL_OBJECT_PREFIX}/channels/"
    if not channel_key.startswith(expected_prefix) or not channel_key.endswith(".json"):
        raise ValueError("model release channel key is invalid")
    try:
        payload = store.get_object_bytes(bucket, channel_key)
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model release channel must be valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("model") != "pp-ocrv6-small":
        raise ValueError("model release channel has an unsupported schema")
    manifest_key = data.get("manifest_key")
    if not isinstance(manifest_key, str) or not manifest_key.startswith(f"{MODEL_OBJECT_PREFIX}/") or not manifest_key.endswith("/manifest.json"):
        raise ValueError("model release channel manifest key is invalid")
    return ModelReleaseChannel(manifest_key=manifest_key)
