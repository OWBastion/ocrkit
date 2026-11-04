from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from training.importer import (
    ScreenshotSetIntegrityError,
    ScreenshotSetMetadata,
    ScreenshotSetNotFinalizedError,
    import_screenshot_set,
)


def _png(width: int = 1280, height: int = 720, shade: int = 200) -> bytes:
    return cv2.imencode(".png", np.full((height, width, 3), shade, dtype=np.uint8))[1].tobytes()


def _member(
    source_id: str = "src-1",
    *,
    data: bytes | None = None,
    object_key: str | None = None,
    layout_version: str = "1280x720-v10",
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
            "layout_version": "1280x720-v10",
            "accuracy": "accurate",
        }
        assert report.screenshot_set["set_id"] == "set-abc"
        assert report.screenshot_set["version"] == 3
        assert report.screenshot_set["finalized_at"] == "2026-08-01T12:00:00Z"
        assert report.screenshot_set["imported_at"]
        assert report.screenshot_set["code_revision"]


    def test_unfinalized_set_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({})
        with pytest.raises(ScreenshotSetNotFinalizedError):
            import_screenshot_set(metadata=_metadata(finalized=False), download_object=download, workspace=tmp_path / "set")


    def test_layout_mismatch_fails(self, tmp_path: Path) -> None:
        download, _ = _downloader({"evidence/screens/src-1.png": _png()})
        with pytest.raises(ScreenshotSetIntegrityError, match="src-1"):
            import_screenshot_set(
                metadata=_metadata([_member("src-1", layout_version="1280x800-v4")]),
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
