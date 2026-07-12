from __future__ import annotations

import importlib.util
import sys
import datetime as dt
from pathlib import Path


def _load_module():
    path = Path("training/scripts/next_model_version.py")
    spec = importlib.util.spec_from_file_location("next_model_version", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubClient:
    def __init__(self, occupied: set[str]) -> None:
        self.occupied = occupied

    def head_object(self, *, Bucket: str, Key: str) -> None:
        if Key.rsplit("/", 1)[0] not in self.occupied:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")


def test_next_model_version_adds_collision_suffix(monkeypatch, capsys) -> None:
    module = _load_module()
    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 12, 9, 30, 15, tzinfo=tz)

    monkeypatch.setattr(module.dt, "datetime", FixedDateTime)
    client = StubClient({"models/pp-ocrv6-small/2026.07.12-093015"})
    monkeypatch.setattr(module.boto3, "client", lambda *_args, **_kwargs: client)
    monkeypatch.setenv("OCRKIT_R2_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("OCRKIT_R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("OCRKIT_R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(sys, "argv", ["next_model_version.py", "--bucket", "bucket"])

    module.main()

    assert capsys.readouterr().out.strip() == "2026.07.12-093015-01"
