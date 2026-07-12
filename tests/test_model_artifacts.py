from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.model_artifacts import ModelArtifactError, ModelArtifactStore


class StubObjectStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_object_bytes(self, bucket: str, object_key: str, version_id: str | None = None) -> bytes:
        return self.objects[object_key]


def _manifest(version: str, files: dict[str, bytes]) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "model": "pp-ocrv6-small",
            "version": version,
            "files": {
                name: {
                    "object_key": f"models/pp-ocrv6-small/{version}/{name}",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in files.items()
            },
        }
    ).encode()


def test_prepare_downloads_and_reuses_verified_model(tmp_path: Path) -> None:
    files = {
        "det.onnx": b"det",
        "rec.onnx": b"rec",
        "rec_dict.txt": b"a\nb\n",
        "rapidocr.yaml": b"Det: {}\nRec: {}\n",
    }
    objects = {f"models/pp-ocrv6-small/v1/{name}": payload for name, payload in files.items()}
    objects["models/pp-ocrv6-small/v1/manifest.json"] = _manifest("v1", files)
    store = ModelArtifactStore(StubObjectStore(objects), "model-bucket", tmp_path)

    artifacts = store.prepare("models/pp-ocrv6-small/v1/manifest.json")
    assert artifacts.version == "v1"
    assert artifacts.rapidocr_config_path.read_bytes() == files["rapidocr.yaml"]

    artifacts = store.prepare("models/pp-ocrv6-small/v1/manifest.json")
    assert artifacts.rapidocr_config_path.parent == tmp_path / "v1"


def test_prepare_rejects_bad_checksum(tmp_path: Path) -> None:
    files = {
        "det.onnx": b"det",
        "rec.onnx": b"rec",
        "rec_dict.txt": b"a\n",
        "rapidocr.yaml": b"Det: {}\nRec: {}\n",
    }
    manifest = json.loads(_manifest("v1", files))
    manifest["files"]["rec.onnx"]["sha256"] = "0" * 64
    objects = {f"models/pp-ocrv6-small/v1/{name}": payload for name, payload in files.items()}
    objects["models/pp-ocrv6-small/v1/manifest.json"] = json.dumps(manifest).encode()

    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        ModelArtifactStore(StubObjectStore(objects), "model-bucket", tmp_path).prepare("models/pp-ocrv6-small/v1/manifest.json")


def test_prepare_rejects_path_traversal(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "version": "v1",
        "files": {
            name: {"object_key": f"models/pp-ocrv6-small/v1/{name}", "sha256": "0" * 64, "size_bytes": 1}
            for name in {"det.onnx", "rec.onnx", "rec_dict.txt", "rapidocr.yaml"}
        },
    }
    manifest["files"]["det.onnx"]["object_key"] = "../det.onnx"

    with pytest.raises(ModelArtifactError, match="object key"):
        ModelArtifactStore(
            StubObjectStore({"models/pp-ocrv6-small/v1/manifest.json": json.dumps(manifest).encode()}),
            "model-bucket",
            tmp_path,
        ).prepare(
            "models/pp-ocrv6-small/v1/manifest.json"
        )


def test_prepare_rejects_cross_bucket_prefix_reference(tmp_path: Path) -> None:
    files = {name: b"x" for name in {"det.onnx", "rec.onnx", "rec_dict.txt", "rapidocr.yaml"}}
    manifest = json.loads(_manifest("v1", files))
    manifest["files"]["det.onnx"]["object_key"] = "uploads/screenshot.png"

    with pytest.raises(ModelArtifactError, match="outside"):
        ModelArtifactStore(
            StubObjectStore({"models/pp-ocrv6-small/v1/manifest.json": json.dumps(manifest).encode()}),
            "model-bucket",
            tmp_path,
        ).prepare("models/pp-ocrv6-small/v1/manifest.json")
