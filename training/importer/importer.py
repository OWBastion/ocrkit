"""Offline platform snapshot importer (issue #5).

Consumes one finalized platform-reviewed dataset snapshot and materializes the
source evidence OCRKit needs for its existing preparation/training workflow:

.. code-block:: text

    platform reviewed snapshot
    → local immutable import/materialization
    → source-level train/holdout split
    → Rust ROI crop/export
    → exact transcription materialization
    → validation

The importer never writes to the remote snapshot, never requires platform DB
access or broad R2 credentials, and never manufactures OCR labels from canonical
business values: rec labels come only from reviewed ``exact_transcription``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.roi_config import RoiConfig, load_roi_config
from app.image.roi import crop_all_rois
from training.importer.client import (
    ObjectUnavailableError,
    SnapshotClient,
    SnapshotContractError,
    SnapshotNotFinalizedError,
)
from training.importer.contract import (
    AnnotationsPayload,
    SnapshotMetadata,
)
from training.importer.split import split_rule_parameters, split_sources
from training.scripts.validate_annotations import validate_rec

IMPORTER_VERSION = "1"
SUPPORTED_IMAGE_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}
DEFAULT_SPLIT_SEED = "ocrkit-v1"
DEFAULT_HOLDOUT_FRACTION = 0.2


class MissingSourceError(RuntimeError):
    """A referenced snapshot member object is unavailable; nothing is substituted."""


class SnapshotIntegrityError(RuntimeError):
    """Downloaded bytes do not match the snapshot's declared checksum."""


@dataclass
class ImportReport:
    snapshot_id: str
    snapshot_version: str
    sources: int
    annotations: int
    split: dict[str, int]
    labels: dict[str, int]
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output: str = ""
    workspace: str = ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _mime_extension(mime_type: str) -> str:
    extension = SUPPORTED_IMAGE_MIME.get(mime_type)
    if extension is None:
        raise SnapshotContractError(f"unsupported evidence mime type: {mime_type}")
    return extension


def _code_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _verify_annotations(metadata: SnapshotMetadata, payload: AnnotationsPayload) -> None:
    if payload.snapshot_id != metadata.snapshot_id:
        raise SnapshotContractError(
            f"annotation payload belongs to {payload.snapshot_id!r}, not {metadata.snapshot_id!r}"
        )
    sources_by_id = {obj.source_id: obj for obj in metadata.sources}
    crop_object_ids = {obj.object_id for obj in metadata.objects if obj.kind == "crop"}
    for ann in payload.annotations:
        source = sources_by_id.get(ann.source_id)
        if source is None:
            raise SnapshotContractError(f"annotation {ann.annotation_id} references unknown source {ann.source_id}")
        if source.layout_version is None:
            raise SnapshotContractError(f"source {ann.source_id} is missing a layout_version")
        if ann.layout_version != source.layout_version:
            raise SnapshotContractError(
                f"annotation {ann.annotation_id} layout {ann.layout_version} does not match its source {source.layout_version}"
            )
        if ann.crop_object_id is not None and ann.crop_object_id not in crop_object_ids:
            raise SnapshotContractError(f"annotation {ann.annotation_id} references unknown crop object {ann.crop_object_id}")


def _download_objects(
    client: SnapshotClient,
    objects: list[Any],
    workspace: Path,
    resume: bool,
) -> dict[str, Path]:
    """Download and checksum-verify snapshot members; resume reuses verified files."""
    target_dir = workspace / "objects"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for obj in objects:
        extension = _mime_extension(obj.mime_type)
        name = obj.source_id if obj.kind == "source" else f"crop-{obj.annotation_id}"
        target = target_dir / f"{name}{extension}"
        if resume and target.is_file() and _sha256_file(target) == obj.sha256:
            paths[obj.object_id] = target
            continue
        try:
            data = client.download_object(obj.object_id)
        except ObjectUnavailableError as exc:
            raise MissingSourceError(f"snapshot evidence unavailable: {obj.object_id}") from exc
        if _sha256_bytes(data) != obj.sha256:
            raise SnapshotIntegrityError(f"object {obj.object_id} checksum mismatch")
        target.write_bytes(data)
        paths[obj.object_id] = target
    return paths


def _rust_cli_command() -> list[str]:
    configured = os.environ.get("OCRKIT_RUST_IMAGE_CLI")
    if configured:
        return [configured]
    return [
        "cargo",
        "run",
        "--manifest-path",
        str(ROOT / "rust/Cargo.toml"),
        "--locked",
        "-p",
        "ocrkit-image-cli",
        "--quiet",
        "--",
    ]


def _rust_crop_sources(
    sources_by_layout: dict[str, list[tuple[Any, str]]],
    workspace: Path,
    layout_config_paths: dict[str, Path],
) -> Path:
    """Run the existing Rust ROI crop/export per layout and merge manifests."""
    merged_sources: list[dict[str, Any]] = []
    layout_versions: list[str] = []
    for layout_version, sources in sources_by_layout.items():
        roi_config_path = layout_config_paths.get(layout_version)
        if roi_config_path is None:
            raise SnapshotContractError(f"no ROI config for layout version {layout_version}")
        group_out = workspace / f".rust-crops-{layout_version}"
        group_out.mkdir(parents=True, exist_ok=True)
        layout_manifest = workspace / f"layout-manifest-{layout_version}.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_layout_manifest.py"),
                str(roi_config_path),
                str(layout_manifest),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        cases_path = workspace / f"cases-{layout_version}.json"
        cases_path.write_text(
            json.dumps(
                [
                    {"id": source.source_id, "image": f"objects/{source.source_id}{_mime_extension(source.mime_type)}", "split": source_split}
                    for source, source_split in sources
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [
                    *_rust_cli_command(),
                    "crop-batch",
                    "--manifest",
                    str(layout_manifest),
                    "--cases",
                    str(cases_path),
                    "--input-root",
                    str(workspace),
                    "--output-dir",
                    str(group_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown Rust crop error").strip()
            raise RuntimeError(f"Rust ROI crop failed for {layout_version}: {detail}") from exc

        group_manifest = json.loads((group_out / "crop_manifest.json").read_text(encoding="utf-8"))
        merged_sources.extend(group_manifest.get("sources", []))
        layout_versions.append(layout_version)

    merged = Path(workspace / "merged-crop-manifest.json")
    merged.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "layout_versions": sorted(layout_versions),
                "sources": merged_sources,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return merged


def _line_crop(image: np.ndarray, box: list[list[float]]) -> np.ndarray | None:
    """Crop a four-point box from an image (mirrors prepare_rec_candidates._crop_line)."""
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        return None
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(image.shape[1], int(np.ceil(points[:, 0].max())))
    y2 = min(image.shape[0], int(np.ceil(points[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def _normalized_sources(
    workspace: Path,
    metadata: SnapshotMetadata,
    layout_configs: dict[str, RoiConfig],
) -> dict[str, np.ndarray]:
    """Decode and normalize each source screenshot for line-crop derivation."""
    normalized: dict[str, np.ndarray] = {}
    for obj in metadata.sources:
        assert obj.source_id is not None and obj.layout_version is not None
        roi_config = layout_configs.get(obj.layout_version)
        if roi_config is None:
            raise SnapshotContractError(f"no ROI config for layout version {obj.layout_version}")
        path = workspace / "objects" / f"{obj.source_id}{_mime_extension(obj.mime_type)}"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise MissingSourceError(f"source image cannot be decoded: {obj.source_id}")
        normalized[obj.source_id], _ = crop_all_rois(image, roi_config)
    return normalized


def _materialize(
    *,
    metadata: SnapshotMetadata,
    payload: AnnotationsPayload,
    split: dict[str, str],
    workspace: Path,
    output: Path,
    layout_configs: dict[str, RoiConfig],
    needs_rust: bool,
    code_revision: str,
    split_seed: str,
    holdout_fraction: float,
) -> ImportReport:
    source_splits = {source.source_id: split[source.source_id] for source in metadata.sources}
    normalized_sources: dict[str, np.ndarray] | None = None

    if output.exists():
        raise FileExistsError(f"materialized import already exists: {output}")

    annotations_out: list[dict[str, Any]] = []
    label_lines: dict[str, list[str]] = {"train": [], "holdout": []}
    conflicts: list[dict[str, Any]] = []
    # crop_path -> [(annotation_id, transcription, split)]
    per_crop: dict[str, list[tuple[str, str, str]]] = {}

    for ann in payload.annotations:
        source = next(source for source in metadata.sources if source.source_id == ann.source_id)
        ann_split = source_splits[ann.source_id]
        if ann.crop_object_id is not None:
            crop_obj = next(obj for obj in metadata.objects if obj.object_id == ann.crop_object_id)
            crop_path = f"crops/{ann_split}/{ann.source_id}_{ann.annotation_id}{_mime_extension(crop_obj.mime_type)}"
        elif ann.box is not None:
            crop_path = f"crops/{ann_split}/{ann.source_id}_{ann.annotation_id}.png"
        elif ann.roi is not None:
            roi_config = layout_configs.get(source.layout_version or "")
            if roi_config is None or ann.roi not in roi_config.rois:
                raise SnapshotContractError(f"annotation {ann.annotation_id} references unknown ROI {ann.roi!r}")
            crop_path = f"images/{ann_split}/{ann.source_id}/{ann.roi}.png"
        else:  # pragma: no cover - contract validation forbids this state
            raise SnapshotContractError(f"annotation {ann.annotation_id} has no crop source")

        per_crop.setdefault(crop_path, []).append((ann.annotation_id, ann.exact_transcription.strip(), ann_split))
        annotations_out.append(
            {
                "annotation_id": ann.annotation_id,
                "source_id": ann.source_id,
                "split": ann_split,
                "layout_version": ann.layout_version,
                "roi": ann.roi,
                "field": ann.field,
                "ocr_prediction": ann.ocr_prediction,
                "exact_transcription": ann.exact_transcription,
                "canonical_value": ann.canonical_value,
                "crop_path": crop_path,
            }
        )

    # Write everything into a temp dir first, then move it into place atomically.
    tmp_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    tmp_output.mkdir(parents=True)

    try:
        (tmp_output / "labels").mkdir(parents=True)
        for crop_path, entries in per_crop.items():
            distinct = {transcription for _, transcription, _ in entries}
            if len(distinct) > 1:
                conflicts.append(
                    {
                        "crop": crop_path,
                        "transcriptions": sorted(distinct),
                        "annotation_ids": [annotation_id for annotation_id, _, _ in entries],
                    }
                )
                continue
            label_split = entries[0][2]
            label_lines[label_split].append(f"{crop_path}\t{next(iter(distinct))}")

        for label_split in ("train", "holdout"):
            label_lines[label_split].sort()
            (tmp_output / "labels" / f"{label_split}.txt").write_text(
                "\n".join(label_lines[label_split]) + ("\n" if label_lines[label_split] else ""),
                encoding="utf-8",
            )

        warnings: list[str] = []
        if conflicts:
            warnings.append(f"label conflicts excluded from rec labels: {len(conflicts)} crop(s)")

        if needs_rust:
            for layout_dir in sorted(workspace.glob(".rust-crops-*")):
                images = layout_dir / "images"
                if not images.is_dir():
                    continue
                for path in sorted(images.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(images)
                    destination = tmp_output / "images" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)

        # Line and pre-crop samples.
        crop_objects = {obj.object_id: obj for obj in metadata.objects if obj.kind == "crop"}
        for ann in payload.annotations:
            source = next(source for source in metadata.sources if source.source_id == ann.source_id)
            ann_split = source_splits[ann.source_id]
            if ann.crop_object_id is not None:
                crop_obj = crop_objects[ann.crop_object_id]
                source_path = workspace / "objects" / f"crop-{ann.annotation_id}{_mime_extension(crop_obj.mime_type)}"
                destination = tmp_output / "crops" / ann_split / f"{ann.source_id}_{ann.annotation_id}{_mime_extension(crop_obj.mime_type)}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            elif ann.box is not None:
                if normalized_sources is None:
                    normalized_sources = _normalized_sources(workspace, metadata, layout_configs)
                source_image = normalized_sources[ann.source_id]
                crop = _line_crop(source_image, ann.box)
                if crop is None:
                    raise SnapshotContractError(f"annotation {ann.annotation_id} box is outside the normalized source")
                destination = tmp_output / "crops" / ann_split / f"{ann.source_id}_{ann.annotation_id}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), crop):
                    raise RuntimeError(f"failed to write line crop: {destination}")

        (tmp_output / "annotations.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in annotations_out) + "\n",
            encoding="utf-8",
        )
        (tmp_output / "snapshot.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (tmp_output / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "importer_version": IMPORTER_VERSION,
                    "snapshot": {
                        "snapshot_id": metadata.snapshot_id,
                        "version": metadata.version,
                        "finalized_at": metadata.finalized_at,
                    },
                    "split": {
                        **split_rule_parameters(split_seed, holdout_fraction),
                        "assignment": dict(sorted(split.items())),
                    },
                    "layout_versions": sorted({source.layout_version for source in metadata.sources if source.layout_version}),
                    "code_revision": code_revision,
                    "imported_at": datetime.now(UTC).isoformat(),
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "object_id": source.object_id,
                            "sha256": source.sha256,
                            "size_bytes": source.size_bytes,
                            "layout_version": source.layout_version,
                        }
                        for source in metadata.sources
                    ],
                    "annotation_count": len(payload.annotations),
                    "labels": {label_split: len(label_lines[label_split]) for label_split in ("train", "holdout")},
                    "warnings": warnings,
                    "label_conflicts": conflicts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        merged_manifest = workspace / "merged-crop-manifest.json"
        if needs_rust and merged_manifest.is_file():
            shutil.copy2(merged_manifest, tmp_output / "crop_manifest.json")
        else:
            (tmp_output / "crop_manifest.json").write_text(
                json.dumps({"schema_version": "1", "layout_versions": [], "sources": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        # Validate the materialized rec labels with the existing validator.
        for label_split in ("train", "holdout"):
            label_file = tmp_output / "labels" / f"{label_split}.txt"
            validate_rec(label_file)
    except Exception:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise

    tmp_output.rename(output)

    return ImportReport(
        snapshot_id=metadata.snapshot_id,
        snapshot_version=metadata.version,
        sources=len(metadata.sources),
        annotations=len(payload.annotations),
        split={"train": sum(1 for value in split.values() if value == "train"), "holdout": sum(1 for value in split.values() if value == "holdout")},
        labels={"train": len(label_lines["train"]), "holdout": len(label_lines["holdout"])},
        conflicts=conflicts,
        warnings=warnings,
        output=str(output),
        workspace=str(workspace),
    )


def default_layout_configs() -> dict[str, tuple[RoiConfig, Path]]:
    """Load the supported layout/ROI configs keyed by layout version."""
    configs: dict[str, tuple[RoiConfig, Path]] = {}
    for path in (ROOT / "configs/roi_1280x720.yaml", ROOT / "configs/roi_1280x800.yaml"):
        config = load_roi_config(path)
        configs[config.version] = (config, path)
    return configs


def import_snapshot(
    *,
    client: SnapshotClient,
    snapshot_id: str,
    workspace: Path,
    output: Path,
    layout_configs: dict[str, tuple[RoiConfig, Path]] | None = None,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    split_seed: str = DEFAULT_SPLIT_SEED,
    resume: bool = True,
    code_revision: str | None = None,
) -> ImportReport:
    """Import one finalized snapshot into a materialized local dataset.

    ``workspace`` caches verified downloads and is safe to reuse for resume;
    ``output`` is written once (refusing overwrite) and never contains
    credentials, object URLs, or remote snapshot writes.
    """
    layout_configs = layout_configs or default_layout_configs()
    metadata = client.fetch_snapshot(snapshot_id)
    if metadata.snapshot_id != snapshot_id:
        raise SnapshotContractError(f"snapshot endpoint returned {metadata.snapshot_id!r}, expected {snapshot_id!r}")
    if not metadata.finalized:
        raise SnapshotNotFinalizedError(f"snapshot {snapshot_id} is not finalized")
    payload = client.fetch_annotations(snapshot_id)
    _verify_annotations(metadata, payload)

    workspace.mkdir(parents=True, exist_ok=True)
    _download_objects(client, metadata.objects, workspace, resume)

    split = split_sources(
        [source.source_id for source in metadata.sources if source.source_id],
        metadata.snapshot_id,
        holdout_fraction,
        split_seed,
    )
    (workspace / "snapshot.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (workspace / "split.json").write_text(
        json.dumps({"snapshot_id": metadata.snapshot_id, **split_rule_parameters(split_seed, holdout_fraction), "assignment": split}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # Rust ROI crops are needed only for field-level annotations without a box.
    needs_rust = any(ann.box is None and ann.crop_object_id is None for ann in payload.annotations)
    if needs_rust:
        sources_by_layout: dict[str, list[tuple[Any, str]]] = {}
        for source in metadata.sources:
            if source.layout_version is None:
                raise SnapshotContractError(f"source {source.source_id} is missing a layout_version")
            sources_by_layout.setdefault(source.layout_version, []).append((source, split[source.source_id]))
        _rust_crop_sources(sources_by_layout, workspace, {version: path for version, (_, path) in layout_configs.items()})

    return _materialize(
        metadata=metadata,
        payload=payload,
        split=split,
        workspace=workspace,
        output=output,
        layout_configs={version: config for version, (config, _) in layout_configs.items()},
        needs_rust=needs_rust,
        code_revision=code_revision or _code_revision(),
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
    )
