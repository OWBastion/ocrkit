"""Materialize one finalized platform screenshot set for Studio review (issue #26).

One finalized screenshot set -> a verified workspace of source images plus the
provenance needed to create a Studio batch:

1. The caller fetches ``ScreenshotSetMetadata`` through the private contract
   (see ``client.py``) — set membership is immutable and is the platform's
   explicit approval to use these sources as OCR training data.
2. Each member is downloaded from R2 by ``object_key`` through a caller-provided
   ``download_object`` callable (a read-only, prefix-scoped R2 key in Studio).
   SHA-256, ``size_bytes`` and ``mime_type`` are verified against the
   platform-signed metadata; a mismatch fails the whole import — no
   substitution, no skipping.
3. The decoded image's ROI layout is re-detected and must match the declared
   ``layout_version`` so Studio crops each source with the config the platform
   used.
4. The import is resumable: verified workspace files are reused; missing files
   are downloaded; corrupt leftovers abort with a clear error.
5. Provenance for every source (set id/version, ``source_id``, object key,
   sha256, layout version, accuracy mark) is returned for ``batch.json`` so the
   Studio review -> labels -> train -> publish path records where each source
   came from. ``accuracy`` stays provenance — a review-ordering hint, never a
   transcription or label.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from app.core.roi_config import RoiConfig, load_roi_config
from app.image.loader import decode_image
from app.image.roi import select_roi_config

from .client import ScreenshotSetNotFinalizedError
from .contract import ScreenshotSetMember, ScreenshotSetMetadata

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_CONFIG_PATHS = (ROOT / "configs/roi_1280x720.yaml", ROOT / "configs/roi_1280x800.yaml")
OBJECTS_DIRNAME = "objects"
SUPPORTED_IMAGE_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
MAX_MEMBER_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_SET_MEMBERS = 512


class ScreenshotSetIntegrityError(RuntimeError):
    """A set member failed checksum, size, layout, or download verification."""


@dataclass(frozen=True)
class ScreenshotSetImport:
    """Everything needed to create a Studio batch from a screenshot set."""

    set_id: str
    version: int
    finalized_at: str
    member_files: list[Path]
    provenance_by_digest: dict[str, dict[str, Any]]
    screenshot_set: dict[str, Any]
    member_count: int
    accuracy_counts: dict[str, int] = field(default_factory=dict)
    workspace: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_filename(index: int, member: ScreenshotSetMember) -> str:
    ext = SUPPORTED_IMAGE_MIME.get(member.mime_type)
    if ext is None:
        raise ScreenshotSetIntegrityError(
            f"screenshot-set member {member.source_id} has unsupported mime_type {member.mime_type!r}"
        )
    return f"{index:04d}-{member.source_id}{ext}"


def _write_member_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename keeps a prior completed download intact on failure.
    temp_path = path.with_suffix(path.suffix + ".part")
    temp_path.write_bytes(data)
    temp_path.replace(path)


def _code_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True
        )
        value = result.stdout.strip()
        return value or "unknown"
    except Exception:
        return "unknown"


def default_layout_configs() -> dict[str, tuple[RoiConfig, Path]]:
    """Layout versions Studio can crop, keyed by declared ``layout_version``."""
    configs: dict[str, tuple[RoiConfig, Path]] = {}
    for path in LAYOUT_CONFIG_PATHS:
        if not path.is_file():
            continue
        config = load_roi_config(path)
        configs[config.version] = (config, path)
    return configs


def import_screenshot_set(
    *,
    metadata: ScreenshotSetMetadata,
    download_object: Callable[[str], bytes],
    workspace: Path,
    expected_version: int | None = None,
    layout_configs: dict[str, tuple[RoiConfig, Path]] | None = None,
    resume: bool = True,
    max_members: int = DEFAULT_MAX_SET_MEMBERS,
    code_revision: str | None = None,
) -> ScreenshotSetImport:
    """Download and verify every member of one finalized screenshot set.

    ``download_object`` receives an R2 object key and returns the object's
    bytes; the caller supplies a read-only, prefix-scoped accessor. Downloaded
    files land under ``workspace/objects/`` so an interrupted import can resume
    against it without re-downloading verified members.
    """
    if expected_version is not None and metadata.version != expected_version:
        raise ScreenshotSetIntegrityError(
            f"screenshot-set version mismatch: requested {expected_version}, metadata reports {metadata.version}"
        )
    if not metadata.finalized:
        raise ScreenshotSetNotFinalizedError(f"screenshot set {metadata.set_id}@{metadata.version} is not finalized")
    if len(metadata.members) > max_members:
        raise ScreenshotSetIntegrityError(
            f"screenshot set has {len(metadata.members)} members; limit is {max_members}"
        )
    if not resume and workspace.exists():
        shutil.rmtree(workspace)
    configs = layout_configs if layout_configs is not None else default_layout_configs()
    if not configs:
        raise ScreenshotSetIntegrityError("no ROI layout configs available to verify screenshot-set members")
    for member in metadata.members:
        if member.layout_version not in configs:
            raise ScreenshotSetIntegrityError(
                f"screenshot-set member {member.source_id} declares unsupported layout_version "
                f"{member.layout_version!r}; Studio supports {sorted(configs)}"
            )
    objects_dir = workspace / OBJECTS_DIRNAME
    objects_dir.mkdir(parents=True, exist_ok=True)
    member_files: list[Path] = []
    provenance_by_digest: dict[str, dict[str, Any]] = {}
    accuracy_counts = {"accurate": 0, "inaccurate": 0, "unset": 0}
    for index, member in enumerate(metadata.members):
        filename = _member_filename(index, member)
        target = objects_dir / filename
        if member.size_bytes > MAX_MEMBER_BYTES:
            raise ScreenshotSetIntegrityError(
                f"screenshot-set member {member.source_id} size {member.size_bytes} exceeds {MAX_MEMBER_BYTES} bytes"
            )
        if target.is_file():
            if target.stat().st_size != member.size_bytes or _sha256_file(target) != member.normalized_sha256:
                raise ScreenshotSetIntegrityError(
                    f"existing file for member {member.source_id} fails checksum; remove {target} to re-download"
                )
        else:
            try:
                data = download_object(member.object_key)
            except Exception as exc:
                raise ScreenshotSetIntegrityError(
                    f"screenshot-set member {member.source_id} could not be downloaded from object storage: {exc}"
                ) from exc
            if len(data) != member.size_bytes or hashlib.sha256(data).hexdigest() != member.normalized_sha256:
                raise ScreenshotSetIntegrityError(f"screenshot-set member {member.source_id} failed checksum verification")
            _write_member_file(target, data)
        member_files.append(target)

        try:
            image = decode_image(target.read_bytes())
            detected = select_roi_config(image, tuple(config for config, _ in configs.values()))
        except ScreenshotSetIntegrityError:
            raise
        except Exception as exc:
            raise ScreenshotSetIntegrityError(
                f"screenshot-set member {member.source_id} does not decode as a supported screenshot: {exc}"
            ) from exc
        if detected.version != member.layout_version:
            raise ScreenshotSetIntegrityError(
                f"screenshot-set member {member.source_id} declares layout {member.layout_version!r} "
                f"but decodes as {detected.version!r}"
            )
        accuracy = member.accuracy
        accuracy_counts[accuracy if accuracy is not None else "unset"] += 1
        provenance_by_digest[member.normalized_sha256] = {
            "source": "screenshot_set",
            "set_id": metadata.set_id,
            "set_version": metadata.version,
            "source_id": member.source_id,
            "object_key": member.object_key,
            "sha256": member.normalized_sha256,
            "layout_version": member.layout_version,
            "accuracy": member.accuracy,
        }

    screenshot_set = {
        "set_id": metadata.set_id,
        "version": metadata.version,
        "finalized_at": metadata.finalized_at,
        "imported_at": datetime.now(tz=UTC).isoformat(),
        "code_revision": code_revision if code_revision is not None else _code_revision(),
    }
    return ScreenshotSetImport(
        set_id=metadata.set_id,
        version=metadata.version,
        finalized_at=metadata.finalized_at,
        member_files=member_files,
        provenance_by_digest=provenance_by_digest,
        screenshot_set=screenshot_set,
        member_count=len(member_files),
        accuracy_counts=accuracy_counts,
        workspace=str(workspace),
    )
