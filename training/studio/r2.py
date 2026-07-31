from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.core.config import settings
from app.storage.r2_client import (
    ObjectAccessDeniedError,
    ObjectListError,
    ObjectNotFoundError,
    ObjectTimeoutError,
    ObjectTooLargeError,
    R2ObjectStore,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DownloadedRemoteImage:
    path: Path
    provenance: dict[str, Any]


@dataclass
class StudioR2Store:
    object_store: R2ObjectStore
    bucket: str
    allowed_prefixes: tuple[str, ...]
    max_objects: int
    max_object_bytes: int

    @classmethod
    def from_settings(cls) -> "StudioR2Store | None":
        if not all((settings.r2_endpoint_url, settings.r2_access_key_id, settings.r2_secret_access_key)):
            return None
        prefixes = tuple(
            prefix.strip()
            for prefix in settings.studio_r2_allowed_prefixes.split(",")
            if prefix.strip()
        )
        if not settings.studio_r2_bucket or not prefixes:
            return None
        object_store = R2ObjectStore.from_settings(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region_name,
            default_bucket=settings.studio_r2_bucket,
            allowed_buckets_raw=settings.studio_r2_bucket,
            read_timeout_seconds=settings.r2_read_timeout_seconds,
        )
        return cls(
            object_store=object_store,
            bucket=settings.studio_r2_bucket,
            allowed_prefixes=prefixes,
            max_objects=settings.studio_r2_max_objects,
            max_object_bytes=settings.studio_r2_max_object_bytes,
        )

    def _validate_prefix(self, prefix: str) -> str:
        normalized = prefix.strip()
        path = PurePosixPath(normalized)
        if not normalized or normalized.startswith("/") or ".." in path.parts:
            raise ObjectAccessDeniedError("Invalid R2 prefix")
        if not any(self._is_under_allowed_prefix(normalized, allowed) for allowed in self.allowed_prefixes):
            raise ObjectAccessDeniedError("R2 prefix is not allowed")
        return normalized

    def _validate_key(self, key: str) -> str:
        normalized = key.strip()
        path = PurePosixPath(normalized)
        if not normalized or normalized.startswith("/") or ".." in path.parts:
            raise ObjectAccessDeniedError("Invalid R2 object key")
        if not any(self._is_under_allowed_prefix(normalized, allowed) for allowed in self.allowed_prefixes):
            raise ObjectAccessDeniedError("R2 object key is not allowed")
        if Path(normalized).suffix.lower() not in _IMAGE_SUFFIXES:
            raise ObjectAccessDeniedError("R2 object is not a supported image")
        return normalized

    @staticmethod
    def _is_under_allowed_prefix(value: str, allowed: str) -> bool:
        root = allowed.rstrip("/")
        return value == root or value.startswith(f"{root}/")

    def list_images(self, prefix: str, continuation_token: str | None = None) -> dict[str, object]:
        validated_prefix = self._validate_prefix(prefix)
        response = self.object_store.list_objects(
            self.bucket,
            validated_prefix,
            continuation_token,
            max_keys=min(self.max_objects, 1000),
        )
        objects: list[dict[str, object]] = []
        for item in response.get("Contents", []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("Key", ""))
            size = int(item.get("Size", 0) or 0)
            if Path(key).suffix.lower() not in _IMAGE_SUFFIXES or size > self.max_object_bytes:
                continue
            last_modified = item.get("LastModified")
            objects.append(
                {
                    "key": key,
                    "size": size,
                    "etag": str(item["ETag"]) if item.get("ETag") else None,
                    "last_modified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else None,
                }
            )
        return {
            "objects": objects,
            "next_cursor": response.get("NextContinuationToken") if response.get("IsTruncated") else None,
        }

    def download_images(self, keys: list[str], temporary_dir: Path) -> list[DownloadedRemoteImage]:
        if not keys:
            raise ValueError("select at least one R2 image")
        unique_keys = list(dict.fromkeys(keys))
        if len(unique_keys) > self.max_objects:
            raise ValueError(f"select at most {self.max_objects} R2 images at a time")
        downloaded: list[DownloadedRemoteImage] = []
        for index, raw_key in enumerate(unique_keys, 1):
            key = self._validate_key(raw_key)
            content = self.object_store.get_object_bytes(
                self.bucket,
                key,
                max_bytes=self.max_object_bytes,
            )
            digest = hashlib.sha256(content).hexdigest()
            suffix = Path(key).suffix.lower()
            path = temporary_dir / f"{index:04d}{suffix}"
            path.write_bytes(content)
            downloaded.append(
                DownloadedRemoteImage(
                    path=path,
                    provenance={
                        "source": "r2",
                        "bucket": self.bucket,
                        "object_key": key,
                        "sha256": digest,
                    },
                )
            )
        return downloaded


def r2_error_detail(exc: Exception) -> str:
    if isinstance(exc, ObjectAccessDeniedError):
        return "R2 bucket 或 prefix 不在 Studio 白名单内"
    if isinstance(exc, ObjectNotFoundError):
        return "R2 图片不存在或已被删除"
    if isinstance(exc, ObjectTooLargeError):
        return "R2 图片超过 Studio 大小限制"
    if isinstance(exc, ObjectTimeoutError):
        return "R2 请求超时，请稍后重试"
    if isinstance(exc, (ObjectListError, ValueError)):
        return str(exc)
    return "R2 读取失败"
