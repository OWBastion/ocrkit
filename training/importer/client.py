"""Snapshot access client (bounded, read-only, no platform DB or R2 credentials).

The HTTP client talks only to the platform's private snapshot contract and the
bounded per-object download path. It never persists credentials, never lists
buckets, and never writes to the remote snapshot.
"""

from __future__ import annotations

import json
from typing import Protocol
from urllib import error as url_error
from urllib import request as url_request

from .contract import AnnotationsPayload, SnapshotMetadata


class SnapshotAuthError(RuntimeError):
    """The platform rejected the snapshot access credentials."""


class SnapshotNotFoundError(RuntimeError):
    """The requested snapshot does not exist or is outside the granted scope."""


class SnapshotNotFinalizedError(RuntimeError):
    """The snapshot exists but is not finalized, so it must not be imported."""


class ObjectUnavailableError(RuntimeError):
    """A member object could not be downloaded (missing, denied, or timed out)."""


class SnapshotContractError(RuntimeError):
    """The platform response did not match the snapshot contract."""


class SnapshotClient(Protocol):
    """Read-only access to one private snapshot and its member evidence."""

    def fetch_snapshot(self, snapshot_id: str) -> SnapshotMetadata: ...

    def fetch_annotations(self, snapshot_id: str) -> AnnotationsPayload: ...

    def download_object(self, object_id: str) -> bytes: ...


class HttpSnapshotClient:
    """urllib-based client for the private snapshot contract.

    ``base_url`` and ``token`` are supplied from environment configuration and
    are never written to any dataset output or log.
    """

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _urlopen(self, req: url_request.Request):
        return url_request.urlopen(req, timeout=self.timeout_seconds)

    def _open(self, path: str) -> bytes:
        req = url_request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with self._urlopen(req) as response:
                return response.read()
        except url_error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SnapshotAuthError(f"snapshot access denied ({exc.code})") from exc
            if exc.code == 404:
                raise SnapshotNotFoundError(f"snapshot member not found: {path}") from exc
            raise SnapshotContractError(f"snapshot endpoint returned HTTP {exc.code}") from exc
        except (url_error.URLError, TimeoutError, OSError) as exc:
            raise SnapshotContractError(f"snapshot endpoint unreachable: {exc}") from exc

    def fetch_snapshot(self, snapshot_id: str) -> SnapshotMetadata:
        try:
            data = json.loads(self._open(f"/api/v1/snapshots/{snapshot_id}"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SnapshotContractError("snapshot metadata is not valid JSON") from exc
        try:
            return SnapshotMetadata.model_validate(data)
        except ValueError as exc:
            raise SnapshotContractError(f"invalid snapshot metadata: {exc}") from exc

    def fetch_annotations(self, snapshot_id: str) -> AnnotationsPayload:
        try:
            data = json.loads(self._open(f"/api/v1/snapshots/{snapshot_id}/annotations"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SnapshotContractError("snapshot annotations are not valid JSON") from exc
        try:
            return AnnotationsPayload.model_validate(data)
        except ValueError as exc:
            raise SnapshotContractError(f"invalid snapshot annotations: {exc}") from exc

    def download_object(self, object_id: str) -> bytes:
        try:
            return self._open(f"/api/v1/objects/{object_id}/download")
        except SnapshotNotFoundError as exc:
            raise ObjectUnavailableError(f"object not available: {object_id}") from exc
        except SnapshotAuthError as exc:
            raise ObjectUnavailableError(f"object download denied: {object_id}") from exc
        except SnapshotContractError as exc:
            raise ObjectUnavailableError(f"object download failed: {object_id}") from exc
