from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path("training/scripts/publish_model_channel.py")
    spec = importlib.util.spec_from_file_location("publish_model_channel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubR2Client:
    def __init__(self) -> None:
        self.puts: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.puts.append(kwargs)


def test_publish_model_channel_writes_candidate_pointer_after_release(monkeypatch) -> None:
    module = _load_module()
    client = StubR2Client()
    monkeypatch.setattr(module.boto3, "client", lambda *_args, **_kwargs: client)
    monkeypatch.setenv("OCRKIT_R2_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("OCRKIT_R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("OCRKIT_R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_model_channel.py",
            "--bucket", "models",
            "--channel-key", "models/pp-ocrv6-small/channels/candidate.json",
            "--manifest-key", "models/pp-ocrv6-small/v2/manifest.json",
        ],
    )

    module.main()

    assert client.puts[0]["Key"] == "models/pp-ocrv6-small/channels/candidate.json"
    assert json.loads(client.puts[0]["Body"].decode())["manifest_key"] == "models/pp-ocrv6-small/v2/manifest.json"


def test_publish_model_channel_rejects_direct_stable_write(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(sys, "argv", [
        "publish_model_channel.py",
        "--bucket", "models",
        "--channel-key", "models/pp-ocrv6-small/channels/stable.json",
        "--manifest-key", "models/pp-ocrv6-small/v2/manifest.json",
    ])

    with pytest.raises(SystemExit, match="explicit promotion"):
        module.main()
