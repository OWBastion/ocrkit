from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib import error as url_error

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from training.importer import (
    HttpScreenshotSetClient,
    ScreenshotSetAuthError,
    ScreenshotSetContractError,
    ScreenshotSetIntegrityError,
    ScreenshotSetMember,
    ScreenshotSetMetadata,
    ScreenshotSetNotFinalizedError,
    ScreenshotSetNotFoundError,
    import_screenshot_set,
)


def _png(width: int = 1280, height: int = 720, shade: int = 200) -> bytes:
    return cv2.imencode(".png", np.full((height, width, 3), shade, dtype=np.uint8))[1].tobytes()


def _member(
    source_id: str = "src-1",
    *,
    data: bytes | None = None,
    object_key: str | None = None,
    layout_version: str = "1280x720-v6",
    accuracy: str | None = None,
    **overrides,
) -> dict[str, object]:
    payload = _png() if data is None else data
    member: dict[str, object] = {
        "source_id": source_id,
        "object_key": object_key or f"evidence/screens/{source_id}.png",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mime_type": "image/png",
        "size_bytes": len(payload),
        "layout_version": layout_version,
        "accuracy": accuracy,
    }
    member.update(overrides)
    return member


def _metadata(members: list[dict[str, object]] | None = None, **overrides) -> ScreenshotSetMetadata:
    data: dict[str, object] = {
        "schema_version": 1,
        "set_id": "set-abc",
        "version": 3,
        "finalized": True,
        "finalized_at": "2026-08-01T12:00:00Z",
        "members": members if members is not None else [_member("src-1")],
    }
    data.update(overrides)
    return ScreenshotSetMetadata.model_validate(data)


def _downloader(objects: dict[str, bytes]) -> tuple:
    calls: list[str] = []

    def _download(key: str) -> bytes:
        calls.append(key)
        if key not in objects:
            raise KeyError(f"missing object {key}")
        return objects[key]

    return _download, calls


class TestContract:
    def test_member_rejects_extra_private_fields(self) -> None:
        with pytest.raises(ValidationError):
            ScreenshotSetMetadata.model_validate(
                {
                    "schema_version": 1,
                    "set_id": "set-abc",
                    "version": 3,
                    "finalized": True,
                    "finalized_at": "2026-08-01T12:00:00Z",
                    "members": [_member("src-1", qq_id="12345", evidence_url="https://private.invalid/x")],
                }
            )

    def test_metadata_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ScreenshotSetMetadata.model_validate(
                {
                    "schema_version": 1,
                    "set_id": "set-abc",
                    "version": 3,
                    "finalized": True,
                    "finalized_at": "2026-08-01T12:00:00Z",
                    "members": [_member("src-1")],
                    "annotations": [{"leak": True}],
                }
            )

    def test_duplicate_source_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate source_id"):
            _metadata([_member("src-1"), _member("src-1", object_key="evidence/screens/other.png", data=_png(shade=10))])

    def test_duplicate_object_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate object_key"):
            _metadata([_member("src-1"), _member("src-2", object_key="evidence/screens/src-1.png", data=_png(shade=10))])

    def test_unsafe_source_id_rejected(self) -> None:
        for bad in ("../escape", "a/b", "a\\b", "a b", ""):
            with pytest.raises(ValidationError):
                ScreenshotSetMember.model_validate(_member(source_id=bad))

    def test_invalid_sha256_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScreenshotSetMember.model_validate(_member("src-1", sha256="not-a-digest"))

    def test_accuracy_accepts_only_known_marks(self) -> None:
        assert _metadata([_member("src-1", accuracy="inaccurate")]).members[0].accuracy == "inaccurate"
        with pytest.raises(ValidationError):
            ScreenshotSetMember.model_validate(_member("src-1", accuracy="wrong"))

    def test_version_must_be_positive_and_schema_fixed(self) -> None:
        with pytest.raises(ValidationError):
            _metadata(version=0)
        with pytest.raises(ValidationError):
            _metadata(schema_version=2)
        with pytest.raises(ValidationError):
            _metadata(members=[])


class TestImportScreenshotSet:
    def test_downloads_verifies_and_returns_provenance(self, tmp_path: Path) -> None:
        one = _png(shade=200)
        two = _png(shade=64)
        members = [
            _member("src-1", data=one, accuracy="accurate"),
            _member("src-2", data=two, accuracy="inaccurate"),
        ]
        objects = {str(member["object_key"]): data for member, data in zip(members, (one, two))}
        download, calls = _downloader(objects)
        metadata = _metadata(members)

        report = import_screenshot_set(metadata=metadata, download_object=download, workspace=tmp_path / "set-3", expected_version=3)

        assert calls == ["evidence/screens/src-1.png", "evidence/screens/src-2.png"]
        assert report.set_id == "set-abc"
        assert report.version == 3
        assert report.member_count == 2
        assert report.accuracy_counts == {"accurate": 1, "inaccurate": 1, "unset": 0}
        assert all(path.is_file() and path.parent == tmp_path / "set-3/objects" for path in report.member_files)
        first = report.provenance_by_digest[hashlib.sha256(one).hexdigest()]
        assert first == {
            "source": "screenshot_set",
            "set_id": "set-abc",
            "set_version": 3,
            "source_id": "src-1",
            "object_key": "evidence/screens/src-1.png",
            "sha256": hashlib.sha256(one).hexdigest(),
            "layout_version": "1280x720-v6",
            "accuracy": "accurate",
        }
        assert report.screenshot_set["set_id"] == "set-abc"
        assert report.screenshot_set["version"] == 3
        assert report.screenshot_set["finalized_at"] == "2026-08-01T12:00:00Z"
        assert report.screenshot_set["imported_at"]
        assert report.screenshot_set["code_revision"]

    def test_version_mismatch_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({})
        with pytest.raises(ScreenshotSetIntegrityError, match="version mismatch"):
            import_screenshot_set(metadata=_metadata(), download_object=download, workspace=tmp_path / "set", expected_version=9)

    def test_unfinalized_set_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({})
        with pytest.raises(ScreenshotSetNotFinalizedError):
            import_screenshot_set(metadata=_metadata(finalized=False), download_object=download, workspace=tmp_path / "set")

    def test_member_limit_enforced(self, tmp_path: Path) -> None:
        download, _ = _downloader({})
        with pytest.raises(ScreenshotSetIntegrityError, match="members"):
            import_screenshot_set(metadata=_metadata(), download_object=download, workspace=tmp_path / "set", max_members=0)

    def test_unsupported_declared_layout_fails_before_download(self, tmp_path: Path) -> None:
        download, calls = _downloader({"evidence/screens/src-1.png": _png()})
        with pytest.raises(ScreenshotSetIntegrityError, match="unsupported layout_version"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", layout_version="9999x9999-v9")]),
                download_object=download,
                workspace=tmp_path / "set",
            )
        assert calls == []

    def test_layout_mismatch_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({"evidence/screens/src-1.png": _png()})
        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", layout_version="1280x800-v1")]),
                download_object=download,
                workspace=tmp_path / "set",
            )

    def test_checksum_mismatch_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({"evidence/screens/src-1.png": _png()})
        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", sha256="0" * 64)]),
                download_object=download,
                workspace=tmp_path / "set",
            )

    def test_missing_object_fails_with_member_identity(self, tmp_path: Path) -> None:
        download, _ = _downloader({})
        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(metadata=_metadata(), download_object=download, workspace=tmp_path / "set")

    def test_oversized_member_fails(self, tmp_path: Path) -> None:
        download, calls = _downloader({"evidence/screens/src-1.png": _png()})
        with pytest.raises(ScreenshotSetIntegrityError, match="exceeds"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", size_bytes=40 * 1024 * 1024)]),
                download_object=download,
                workspace=tmp_path / "set",
            )
        assert calls == []

    def test_undecodable_member_fails(self, tmp_path: Path) -> None:
        payload = b"not-an-image-payload"
        download, _ = _downloader({"evidence/screens/src-1.png": payload})
        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", data=payload)]),
                download_object=download,
                workspace=tmp_path / "set",
            )

    def test_resume_reuses_verified_files(self, tmp_path: Path) -> None:
        data = _png()
        objects = {"evidence/screens/src-1.png": data}
        download, calls = _downloader(objects)
        metadata = _metadata()
        workspace = tmp_path / "set-3"

        import_screenshot_set(metadata=metadata, download_object=download, workspace=workspace)
        assert calls == ["evidence/screens/src-1.png"]

        second, second_calls = _downloader(objects)
        report = import_screenshot_set(metadata=metadata, download_object=second, workspace=workspace)
        assert second_calls == []
        assert report.member_count == 1

    def test_corrupt_workspace_file_fails(self, tmp_path: Path) -> None:
        data = _png()
        objects = {"evidence/screens/src-1.png": data}
        download, _ = _downloader(objects)
        metadata = _metadata()
        workspace = tmp_path / "set-3"
        report = import_screenshot_set(metadata=metadata, download_object=download, workspace=workspace)
        report.member_files[0].write_bytes(b"corrupted")

        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(metadata=metadata, download_object=download, workspace=workspace)

    def test_no_resume_redownloads(self, tmp_path: Path) -> None:
        data = _png()
        objects = {"evidence/screens/src-1.png": data}
        download, _ = _downloader(objects)
        metadata = _metadata()
        workspace = tmp_path / "set-3"
        import_screenshot_set(metadata=metadata, download_object=download, workspace=workspace)

        second, second_calls = _downloader(objects)
        import_screenshot_set(metadata=metadata, download_object=second, workspace=workspace, resume=False)
        assert second_calls == ["evidence/screens/src-1.png"]


class TestHttpScreenshotSetClient:
    class _StubbedClient(HttpScreenshotSetClient):
        def __init__(self, error: Exception | None = None, body: bytes = b"{}") -> None:
            super().__init__("https://example.test", "token")
            self.error = error
            self.body = body
            self.request: object | None = None

        def _urlopen(self, req):
            self.request = req
            if self.error is not None:
                raise self.error
            body = self.body
            return type("Response", (), {"__enter__": lambda self: self, "__exit__": lambda *a: None, "read": lambda self: body})()

    def test_fetches_versioned_set_endpoint_with_bearer(self) -> None:
        client = self._StubbedClient(body=json.dumps({
            "schema_version": 1,
            "set_id": "set-abc",
            "version": 7,
            "finalized": True,
            "finalized_at": "2026-08-01T12:00:00Z",
            "members": [_member("src-1")],
        }).encode())
        metadata = client.fetch_set(7)
        assert metadata.version == 7
        request = client.request
        assert request.full_url == "https://example.test/v1/ocrkit/screenshot-sets/7"  # type: ignore[union-attr]
        assert request.headers["Authorization"] == "Bearer token"  # type: ignore[union-attr]

    def test_auth_error_is_mapped(self) -> None:
        client = self._StubbedClient(error=url_error.HTTPError("url", 401, "unauthorized", {}, None))
        with pytest.raises(ScreenshotSetAuthError):
            client.fetch_set(1)

    def test_not_found_is_mapped(self) -> None:
        client = self._StubbedClient(error=url_error.HTTPError("url", 404, "missing", {}, None))
        with pytest.raises(ScreenshotSetNotFoundError):
            client.fetch_set(9)

    def test_not_finalized_is_mapped(self) -> None:
        client = self._StubbedClient(error=url_error.HTTPError("url", 409, "conflict", {}, None))
        with pytest.raises(ScreenshotSetNotFinalizedError):
            client.fetch_set(3)

    def test_other_http_error_is_contract_error(self) -> None:
        client = self._StubbedClient(error=url_error.HTTPError("url", 422, "invalid", {}, None))
        with pytest.raises(ScreenshotSetContractError):
            client.fetch_set(0)

    def test_invalid_metadata_json_is_contract_error(self) -> None:
        client = self._StubbedClient(body=b"not json")
        with pytest.raises(ScreenshotSetContractError, match="not valid JSON"):
            client.fetch_set(1)

    def test_schema_violation_is_contract_error(self) -> None:
        client = self._StubbedClient(body=b'{"schema_version": 1, "set_id": "x", "version": 1, "finalized": true, "finalized_at": "t", "members": []}')
        with pytest.raises(ScreenshotSetContractError, match="invalid screenshot-set metadata"):
            client.fetch_set(1)
