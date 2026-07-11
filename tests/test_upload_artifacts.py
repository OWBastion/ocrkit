from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError


def _load_module():
    path = Path("training/scripts/upload_artifacts.py")
    spec = importlib.util.spec_from_file_location("upload_artifacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubR2Client:
    def __init__(self, existing_keys: set[str]) -> None:
        self.existing_keys = existing_keys
        self.uploads: list[tuple[str, str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> None:
        if Key not in self.existing_keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads.append((filename, bucket, key))


def _artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    files = {
        "det.onnx": b"det",
        "rec.onnx": b"rec",
        "rec_dict.txt": b"a\n",
        "rapidocr.yaml": b"Det: {}\nRec: {}\n",
    }
    for name, payload in files.items():
        (artifact_dir / name).write_bytes(payload)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    name: {"object_key": f"models/pp-ocrv6-small/v1/{name}"} for name in files
                }
            }
        ),
        encoding="utf-8",
    )
    return artifact_dir


def _run_main(monkeypatch: pytest.MonkeyPatch, artifact_dir: Path, client: StubR2Client) -> None:
    module = _load_module()
    monkeypatch.setattr(module.boto3, "client", lambda *_args, **_kwargs: client)
    monkeypatch.setenv("OCRKIT_R2_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("OCRKIT_R2_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("OCRKIT_R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(sys, "argv", ["upload_artifacts.py", "--artifact-dir", str(artifact_dir), "--bucket", "models"])
    module.main()


def test_upload_rejects_existing_version_before_uploading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    client = StubR2Client({"models/pp-ocrv6-small/v1/rec.onnx"})

    with pytest.raises(SystemExit, match="already exists"):
        _run_main(monkeypatch, artifact_dir, client)

    assert client.uploads == []


def test_uploads_all_artifacts_and_manifest_for_new_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    client = StubR2Client(set())

    _run_main(monkeypatch, artifact_dir, client)

    assert {key for _, _, key in client.uploads} == {
        "models/pp-ocrv6-small/v1/det.onnx",
        "models/pp-ocrv6-small/v1/rec.onnx",
        "models/pp-ocrv6-small/v1/rec_dict.txt",
        "models/pp-ocrv6-small/v1/rapidocr.yaml",
        "models/pp-ocrv6-small/v1/manifest.json",
    }
