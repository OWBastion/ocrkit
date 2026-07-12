from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.storage.r2_client import R2ObjectStore
from app.model_artifacts.constants import MODEL_OBJECT_PREFIX, model_version_prefix


class ModelArtifactError(Exception):
    pass


@dataclass(frozen=True)
class ModelArtifacts:
    version: str
    rapidocr_config_path: Path


@dataclass(frozen=True)
class _ArtifactFile:
    object_key: str
    sha256: str
    size_bytes: int


class ModelArtifactStore:
    _required_files = {"det.onnx", "rec.onnx", "rec_dict.txt", "rapidocr.yaml"}

    def __init__(self, object_store: R2ObjectStore, bucket: str, cache_dir: Path) -> None:
        self._object_store = object_store
        self._bucket = bucket
        self._cache_dir = cache_dir

    def prepare(self, manifest_key: str) -> ModelArtifacts:
        if not manifest_key.startswith(MODEL_OBJECT_PREFIX + "/"):
            raise ModelArtifactError(f"Model manifest key is outside {MODEL_OBJECT_PREFIX}")
        manifest = self._load_manifest(manifest_key)
        if manifest.get("schema_version") != 1 or manifest.get("model") != "pp-ocrv6-small":
            raise ModelArtifactError("Model manifest has an unsupported schema")
        version = self._validate_version(manifest.get("version"))
        files = self._parse_files(manifest.get("files"), version)
        target = self._cache_dir / version

        if target.exists():
            self._verify_directory(target, files)
            return ModelArtifacts(version=version, rapidocr_config_path=target / "rapidocr.yaml")

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{version}.", dir=self._cache_dir))
        try:
            for filename, artifact in files.items():
                payload = self._object_store.get_object_bytes(self._bucket, artifact.object_key)
                path = staging / filename
                path.write_bytes(payload)
                self._verify_file(path, artifact.sha256, artifact.size_bytes)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return ModelArtifacts(version=version, rapidocr_config_path=target / "rapidocr.yaml")

    def _load_manifest(self, manifest_key: str) -> dict[str, Any]:
        self._validate_key(manifest_key)
        try:
            payload = self._object_store.get_object_bytes(self._bucket, manifest_key)
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError("Model manifest must be valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ModelArtifactError("Model manifest must be an object")
        return manifest

    def _parse_files(self, raw_files: Any, version: str) -> dict[str, _ArtifactFile]:
        if not isinstance(raw_files, dict) or set(raw_files) != self._required_files:
            raise ModelArtifactError("Model manifest has an invalid file set")

        files: dict[str, _ArtifactFile] = {}
        expected_prefix = model_version_prefix(version) + "/"
        for filename in self._required_files:
            item = raw_files[filename]
            if not isinstance(item, dict):
                raise ModelArtifactError(f"Model manifest entry for {filename} is invalid")
            object_key = item.get("object_key")
            digest = item.get("sha256")
            size_bytes = item.get("size_bytes")
            self._validate_key(object_key)
            if not object_key.startswith(expected_prefix):
                raise ModelArtifactError(f"Model object key is outside {MODEL_OBJECT_PREFIX}")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ModelArtifactError(f"Model manifest hash for {filename} is invalid")
            if not isinstance(size_bytes, int) or size_bytes < 1:
                raise ModelArtifactError(f"Model manifest size for {filename} is invalid")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ModelArtifactError(f"Model manifest hash for {filename} is invalid") from exc
            files[filename] = _ArtifactFile(
                object_key=object_key,
                sha256=digest.lower(),
                size_bytes=size_bytes,
            )
        return files

    @staticmethod
    def _validate_key(key: Any) -> None:
        if not isinstance(key, str) or not key or key.startswith("/") or ".." in Path(key).parts:
            raise ModelArtifactError("Model object key is invalid")

    @staticmethod
    def _validate_version(version: Any) -> str:
        if not isinstance(version, str) or not version or "/" in version or version in {".", ".."}:
            raise ModelArtifactError("Model manifest version is invalid")
        return version

    def _verify_directory(self, directory: Path, files: dict[str, _ArtifactFile]) -> None:
        for filename, artifact in files.items():
            path = directory / filename
            if not path.is_file():
                raise ModelArtifactError(f"Cached model file is missing: {filename}")
            self._verify_file(path, artifact.sha256, artifact.size_bytes)

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str, expected_size: int) -> None:
        if path.stat().st_size != expected_size:
            raise ModelArtifactError(f"Model artifact size mismatch: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ModelArtifactError(f"Model artifact checksum mismatch: {path.name}")
