"""Screenshot-set metadata client (bounded, read-only).

The HTTP client talks only to the platform's private screenshot-set contract.
Member evidence is not downloaded through the API; objects are fetched from R2
by the caller with a read-only, prefix-scoped key. The client never persists
credentials and never writes to the platform.
"""

from __future__ import annotations

import json
from typing import Protocol
from urllib import error as url_error
from urllib import request as url_request

from .contract import ScreenshotSetMetadata


class ScreenshotSetAuthError(RuntimeError):
    """The platform rejected the screenshot-set access credentials."""


class ScreenshotSetNotFoundError(RuntimeError):
    """The requested screenshot set version does not exist."""


class ScreenshotSetNotFinalizedError(RuntimeError):
    """The screenshot set exists but is not finalized, so it must not be imported."""


class ScreenshotSetContractError(RuntimeError):
    """The platform response did not match the screenshot-set contract."""


class ScreenshotSetClient(Protocol):
    """Read-only access to one finalized screenshot set's metadata."""

    def fetch_set(self, version: int) -> ScreenshotSetMetadata: ...


class HttpScreenshotSetClient:
    """urllib-based client for the private screenshot-set contract.

    ``base_url`` and ``token`` are supplied from environment configuration and
    are never written to any dataset output or log.
    """

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _urlopen(self, req: url_request.Request):
        return url_request.urlopen(req, timeout=self.timeout_seconds)

    def fetch_set(self, version: int) -> ScreenshotSetMetadata:
        req = url_request.Request(
            f"{self.base_url}/v1/ocrkit/screenshot-sets/{version}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with self._urlopen(req) as response:
                body = response.read()
        except url_error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ScreenshotSetAuthError(f"screenshot-set access denied ({exc.code})") from exc
            if exc.code == 404:
                raise ScreenshotSetNotFoundError(f"screenshot set not found: version {version}") from exc
            if exc.code == 409:
                raise ScreenshotSetNotFinalizedError(f"screenshot set version {version} is not finalized") from exc
            raise ScreenshotSetContractError(f"screenshot-set endpoint returned HTTP {exc.code}") from exc
        except (url_error.URLError, TimeoutError, OSError) as exc:
            raise ScreenshotSetContractError(f"screenshot-set endpoint unreachable: {exc}") from exc
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ScreenshotSetContractError("screenshot-set metadata is not valid JSON") from exc
        try:
            return ScreenshotSetMetadata.model_validate(data)
        except ValueError as exc:
            raise ScreenshotSetContractError(f"invalid screenshot-set metadata: {exc}") from exc
