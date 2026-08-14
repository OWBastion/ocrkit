from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2

from app.core.roi_config import load_roi_config
from app.image.loader import decode_image
from app.image.quality import assess_input_quality
from app.image.roi import crop_all_rois, select_roi_config
from training.scripts.finalize_rec_labels import finalize
from training.scripts.prepare_rec_candidates import (
    best_rapid_candidate,
    canonicalize,
    candidate_artifact_version,
    candidate_rejection_reason,
    create_artifact_ocr,
    deduplicate_candidate_rows,
    discover_candidate_artifact,
    engine_results_agree,
    load_negative_candidates,
    negative_candidate_rejection_reason,
    prepare_candidates,
)
from training.scripts.validate_annotations import validate_rec
from training.vision import VisionLine, VisionOcr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = ROOT / "training/.work/studio"
DEFAULT_CANDIDATE_ARTIFACT_ROOT = ROOT / "training/.work/artifacts"
DEFAULT_ROI_CONFIG = ROOT / "configs/roi_1280x720.yaml"
ROI_CONFIG_VARIANTS = (DEFAULT_ROI_CONFIG, ROOT / "configs/roi_1280x800.yaml")
PRIVATE_DATASET_ROOT = ROOT / "datasets/labeled/rec/studio"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _invalidate_labels(dataset_dir: Path) -> None:
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "holdout"):
        temporary = labels_dir / f"{split}.txt.{uuid.uuid4().hex}.tmp"
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, labels_dir / f"{split}.txt")


def _merge_crop_manifests(dataset_dir: Path, temporary_dir: Path) -> None:
    temporary_manifest = temporary_dir / "crop_manifest.json"
    if not temporary_manifest.is_file():
        return
    destination = dataset_dir / "crop_manifest.json"
    incoming = json.loads(temporary_manifest.read_text(encoding="utf-8"))
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing["sources"] = [*existing.get("sources", []), *incoming.get("sources", [])]
        _atomic_json(destination, existing)
    else:
        shutil.copy2(temporary_manifest, destination)


def _candidate_artifact() -> tuple[Path, str] | None:
    configured = os.environ.get("OCRKIT_STUDIO_CANDIDATE_ARTIFACT_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path, candidate_artifact_version(path)
    return discover_candidate_artifact(DEFAULT_CANDIDATE_ARTIFACT_ROOT)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _roi_config_options(config_path: Path) -> list[tuple[Path, Any]]:
    """Return the configured layout plus built-in aspect-ratio variants."""
    configured = config_path.resolve()
    paths = [configured]
    if configured == DEFAULT_ROI_CONFIG.resolve():
        paths = [path.resolve() for path in ROI_CONFIG_VARIANTS]
    return [(path, load_roi_config(path)) for path in paths]


def _select_source_roi_config(
    image: Any,
    options: list[tuple[Path, Any]],
) -> tuple[Path, Any]:
    configs = tuple(config for _, config in options)
    selected = select_roi_config(image, configs)
    return next((path, config) for path, config in options if config == selected)


def _relative_roi_config_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _source_roi_config(
    batch_dir: Path,
    source: dict[str, Any],
    options: list[tuple[Path, Any]],
) -> tuple[Path, Any]:
    configured = source.get("roi_config")
    if isinstance(configured, str) and configured:
        path = (ROOT / configured).resolve()
        return path, load_roi_config(path)
    file = source.get("file")
    if not isinstance(file, str) or not file:
        return options[0]
    image_path = batch_dir / file
    if not image_path.is_file():
        return options[0]
    image = decode_image(image_path.read_bytes())
    return _select_source_roi_config(image, options)


def _layout_summary(sources: list[dict[str, Any]]) -> tuple[str, str]:
    layouts = {(str(source["layout_version"]), str(source["roi_config"])) for source in sources}
    if len(layouts) == 1:
        version, path = next(iter(layouts))
        return version, path
    return "mixed", ""


def _materialize_source_layouts(batch_dir: Path, manifest: dict[str, Any]) -> None:
    options = _roi_config_options(DEFAULT_ROI_CONFIG)
    for source in manifest.get("sources", []):
        file = source.get("file")
        image_path = batch_dir / file if isinstance(file, str) and file else None
        if image_path is not None and image_path.is_file():
            image = decode_image(image_path.read_bytes())
            path, config = _select_source_roi_config(image, options)
        else:
            path, config = _source_roi_config(batch_dir, source, options)
        source["layout_version"] = config.version
        source["roi_config"] = _relative_roi_config_path(path)
    manifest["layout_version"], manifest["roi_config"] = _layout_summary(manifest.get("sources", []))
    manifest["roi_configs"] = sorted({str(source["roi_config"]) for source in manifest.get("sources", [])})


def _negative_registry_path(batch_dir: Path) -> Path:
    return batch_dir.parent.parent / "negative-candidates.jsonl"


def rebuild_negative_registry(batch_dir: Path) -> Path:
    """Persist human-rejected candidate signatures for future batches."""
    work_root = batch_dir.parent.parent
    batches_dir = work_root / "batches"
    owners = [path for path in batches_dir.iterdir() if (path / "batch.json").is_file()] if batches_dir.is_dir() else []
    if batch_dir not in owners:
        owners.append(batch_dir)
    negatives: dict[tuple[str, ...], dict[str, Any]] = {}
    for owner in owners:
        review_roots = [owner / "dataset/review"]
        review_roots.extend(path / "dataset/review" for path in (owner / "dataset-revisions").glob("*") if path.is_dir())
        for review_root in review_roots:
            for split in ("train", "holdout"):
                review_path = review_root / f"{split}.jsonl"
                if not review_path.is_file():
                    continue
                rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                for row in rows:
                    if row.get("review_status") != "rejected" or row.get("auto_reject_reason"):
                        continue
                    roi = row.get("roi")
                    if not isinstance(roi, str) or not roi:
                        continue
                    texts = sorted(
                        {
                            canonicalize(str(row.get(name)))
                            for name in ("candidate_text", "rapidocr_text", "vision_text", "teacher_text")
                            if isinstance(row.get(name), str) and str(row.get(name)).strip()
                        }
                    )
                    crop_sha256 = row.get("crop_sha256")
                    raw_roi_sha256 = row.get("raw_roi_sha256")
                    global_box = row.get("global_box")
                    layout_version = row.get("layout_version")
                    if (
                        not texts
                        and not isinstance(crop_sha256, str)
                        and not isinstance(raw_roi_sha256, str)
                    ):
                        continue
                    key = (
                        roi,
                        "\u001f".join(texts),
                        str(crop_sha256 or ""),
                        str(raw_roi_sha256 or ""),
                        json.dumps(global_box, ensure_ascii=False, separators=(",", ":")),
                        str(layout_version or ""),
                    )
                    negatives[key] = {
                        "schema_version": 2,
                        "roi": roi,
                        "texts": texts,
                        "crop_sha256": crop_sha256,
                        "raw_roi_sha256": raw_roi_sha256,
                        "global_box": global_box,
                        "layout_version": layout_version,
                    }
    registry = _negative_registry_path(batch_dir)
    registry.parent.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(registry, (negatives[key] for key in sorted(negatives)))
    return registry


def apply_negative_matches(batch_dir: Path, negative_path: Path) -> int:
    """Remove pending candidates matching a prior human rejection."""
    negatives = load_negative_candidates(negative_path)
    if not negatives:
        return 0
    dataset_dir = batch_dir / "dataset"
    updated: dict[str, list[dict[str, Any]]] = {}
    rejected = 0
    for split in ("train", "holdout"):
        rows = review_rows(batch_dir, split)
        for row in rows:
            if row.get("review_status") != "pending":
                continue
            crop_path = dataset_dir / str(row.get("crop", ""))
            crop_sha256 = _digest(crop_path) if crop_path.is_file() else row.get("crop_sha256")
            reason = negative_candidate_rejection_reason(
                str(row.get("roi", "")),
                (row.get("candidate_text"), row.get("rapidocr_text"), row.get("vision_text"), row.get("teacher_text")),
                crop_sha256=crop_sha256 if isinstance(crop_sha256, str) else None,
                negative_candidates=negatives,
            )
            if reason and reason.startswith("negative_review."):
                row["review_status"] = "rejected"
                row["transcription"] = None
                row["auto_accept_reason"] = None
                row["auto_reject_reason"] = reason
                rejected += 1
        updated[split] = rows
    for split in ("train", "holdout"):
        _atomic_jsonl(dataset_dir / "review" / f"{split}.jsonl", updated[split])
    return rejected


def deduplicate_review_rows(batch_dir: Path) -> int:
    manifest = load_manifest(batch_dir)
    _materialize_source_layouts(batch_dir, manifest)
    options = _roi_config_options(ROOT / str(manifest.get("roi_config") or DEFAULT_ROI_CONFIG.relative_to(ROOT)))
    source_configs = {
        str(source["id"]): _source_roi_config(batch_dir, source, options)
        for source in manifest.get("sources", [])
    }
    dataset_dir = batch_dir / "dataset"
    updated: dict[str, list[dict[str, Any]]] = {}
    deduplicated = 0
    for split in ("train", "holdout"):
        rows = review_rows(batch_dir, split)
        by_config: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
        for row in rows:
            selected = source_configs.get(str(row.get("source_id")))
            if selected is not None:
                path, config = selected
                by_config.setdefault(str(path), (config, []))[1].append(row)
        if not by_config:
            deduplicated += deduplicate_candidate_rows(rows, options[0][1])
        else:
            for config, config_rows in by_config.values():
                deduplicated += deduplicate_candidate_rows(config_rows, config)
        updated[split] = rows
    if deduplicated:
        for split in ("train", "holdout"):
            _atomic_jsonl(dataset_dir / "review" / f"{split}.jsonl", updated[split])
        _invalidate_labels(dataset_dir)
    return deduplicated


def _split_sources(digests: list[str], holdout_ratio: float) -> dict[str, str]:
    if not 0 <= holdout_ratio < 1:
        raise ValueError("holdout ratio must be in [0, 1)")
    ordered = sorted(digests)
    holdout_count = int(round(len(ordered) * holdout_ratio))
    if len(ordered) >= 2 and holdout_ratio > 0:
        holdout_count = max(1, min(len(ordered) - 1, holdout_count))
    return {digest: "holdout" if index < holdout_count else "train" for index, digest in enumerate(ordered)}


def create_batch(
    upload_paths: list[str | Path],
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    roi_config_path: Path = DEFAULT_ROI_CONFIG,
    holdout_ratio: float = 0.2,
    provenance_by_digest: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Copy valid uploads into a private, ignored batch workspace and assign source-level splits."""
    candidates = [Path(item) for item in upload_paths if item]
    if not candidates:
        raise ValueError("select at least one image")
    options = _roi_config_options(roi_config_path)
    prepared: list[tuple[Path, str, dict[str, Any], Path, Any]] = []
    seen: set[str] = set()
    for source in candidates:
        if source.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if not source.is_file():
            continue
        digest = _digest(source)
        if digest in seen:
            continue
        image = decode_image(source.read_bytes())
        selected_path, selected_config = _select_source_roi_config(image, options)
        prepared.append(
            (
                source,
                digest,
                assess_input_quality(image, selected_config.width, selected_config.height),
                selected_path,
                selected_config,
            )
        )
        seen.add(digest)
    if not prepared:
        raise ValueError("no supported, decodable images were selected")

    batch_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    batch_dir = work_root / "batches" / batch_id
    sources_dir = batch_dir / "sources"
    sources_dir.mkdir(parents=True)
    splits = _split_sources([digest for _, digest, *_ in prepared], holdout_ratio)
    sources: list[dict[str, Any]] = []
    for index, (source, digest, quality, selected_path, selected_config) in enumerate(prepared, 1):
        extension = source.suffix.lower()
        target_name = f"{index:04d}-{digest[:12]}{extension}"
        shutil.copy2(source, sources_dir / target_name)
        row: dict[str, Any] = {
            "id": f"source-{digest[:12]}",
            "file": f"sources/{target_name}",
            "sha256": digest,
            "split": splits[digest],
            "original_name": source.name,
            "quality": quality,
            "layout_version": selected_config.version,
            "roi_config": _relative_roi_config_path(selected_path),
        }
        if provenance_by_digest and digest in provenance_by_digest:
            row["provenance"] = provenance_by_digest[digest]
        sources.append(row)
    layout_version, manifest_roi_config = _layout_summary(sources)
    manifest = {
        "schema_version": "1",
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "layout_version": layout_version,
        "roi_config": manifest_roi_config,
        "roi_configs": sorted({str(source["roi_config"]) for source in sources}),
        "holdout_ratio": holdout_ratio,
        "sources": sources,
    }
    _atomic_json(batch_dir / "batch.json", manifest)
    _atomic_json(
        batch_dir / "cases.json",
        [{"id": row["id"], "image": row["file"], "split": row["split"]} for row in sources],
    )
    return batch_dir, batch_summary(batch_dir)


def append_sources(
    batch_dir: Path,
    upload_paths: list[str | Path],
    *,
    provenance_by_digest: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add new source screenshots without changing existing source-level splits."""
    manifest = load_manifest(batch_dir)
    candidates = [Path(item) for item in upload_paths if item]
    if not candidates:
        raise ValueError("select at least one image")
    options = _roi_config_options(ROOT / str(manifest.get("roi_config") or DEFAULT_ROI_CONFIG))
    existing = {str(row["sha256"]) for row in manifest["sources"]}
    prepared: list[tuple[Path, str, dict[str, Any], Path, Any]] = []
    for source in candidates:
        if source.suffix.lower() not in _IMAGE_SUFFIXES or not source.is_file():
            continue
        digest = _digest(source)
        if digest in existing:
            continue
        image = decode_image(source.read_bytes())
        selected_path, selected_config = _select_source_roi_config(image, options)
        prepared.append(
            (
                source,
                digest,
                assess_input_quality(image, selected_config.width, selected_config.height),
                selected_path,
                selected_config,
            )
        )
        existing.add(digest)
    if not prepared:
        raise ValueError("all selected screenshots already exist in this batch or could not be decoded")

    sources: list[dict[str, Any]] = manifest["sources"]
    existing_holdout = sum(row["split"] == "holdout" for row in sources)
    total_sources = len(sources) + len(prepared)
    target_holdout = int(round(total_sources * float(manifest["holdout_ratio"])))
    if total_sources >= 2 and float(manifest["holdout_ratio"]) > 0:
        target_holdout = max(1, min(total_sources - 1, target_holdout))
    new_holdout = max(0, min(len(prepared), target_holdout - existing_holdout))
    sources_dir = batch_dir / "sources"
    next_index = len(sources) + 1
    added: list[dict[str, Any]] = []
    for index, (source, digest, quality, selected_path, selected_config) in enumerate(sorted(prepared, key=lambda row: row[1])):
        extension = source.suffix.lower()
        target_name = f"{next_index + index:04d}-{digest[:12]}{extension}"
        shutil.copy2(source, sources_dir / target_name)
        row: dict[str, Any] = {
            "id": f"source-{digest[:12]}",
            "file": f"sources/{target_name}",
            "sha256": digest,
            "split": "holdout" if index < new_holdout else "train",
            "original_name": source.name,
            "quality": quality,
            "layout_version": selected_config.version,
            "roi_config": _relative_roi_config_path(selected_path),
        }
        if provenance_by_digest and digest in provenance_by_digest:
            row["provenance"] = provenance_by_digest[digest]
        added.append(row)
    manifest["sources"] = [*sources, *added]
    manifest["layout_version"], manifest["roi_config"] = _layout_summary(manifest["sources"])
    manifest["roi_configs"] = sorted({str(source["roi_config"]) for source in manifest["sources"]})
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(batch_dir / "batch.json", manifest)
    _atomic_json(
        batch_dir / "cases.json",
        [{"id": row["id"], "image": row["file"], "split": row["split"]} for row in manifest["sources"]],
    )
    return {"added": len(added), "batch": batch_summary(batch_dir)}


def load_manifest(batch_dir: Path) -> dict[str, Any]:
    return json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))


def batch_summary(batch_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(batch_dir)
    sources = manifest["sources"]
    splits = Counter(row["split"] for row in sources)
    warnings = sum(bool(row["quality"]["warnings"]) for row in sources)
    return {
        "batch_id": manifest["batch_id"],
        "batch_dir": str(batch_dir),
        "sources": len(sources),
        "train_sources": splits["train"],
        "holdout_sources": splits["holdout"],
        "quality_warnings": warnings,
        "layout_version": manifest["layout_version"],
        "roi_configs": manifest.get("roi_configs", [manifest.get("roi_config", "")]),
        "active_dataset_revision": manifest.get("active_dataset_revision"),
    }


def _prepare_candidate_groups(
    batch_dir: Path,
    sources: list[dict[str, Any]],
    output_dir: Path,
    *,
    roi_config_path: Path,
    crop_backend: str,
    teacher_artifact: tuple[Path, str] | None,
    negative_path: Path,
) -> dict[str, Any]:
    """Prepare each source with the ROI layout matching its aspect ratio."""
    options = _roi_config_options(roi_config_path)
    if not sources:
        return prepare_candidates(
            batch_dir / "cases.json",
            batch_dir,
            output_dir,
            options[0][1],
            crop_backend=crop_backend,
            roi_config_path=options[0][0],
            teacher_model_dir=teacher_artifact[0] if teacher_artifact else None,
            teacher_model_version=teacher_artifact[1] if teacher_artifact else None,
            negative_examples_path=negative_path,
        )

    grouped: dict[str, tuple[Path, Any, list[dict[str, Any]]]] = {}
    for source in sources:
        path, config = _source_roi_config(batch_dir, source, options)
        grouped.setdefault(str(path), (path, config, []))[2].append(source)

    temporary_dir = batch_dir / f".candidate-layouts-{uuid.uuid4().hex}"
    temporary_dir.mkdir()
    summaries: list[dict[str, Any]] = []
    try:
        for index, (path, config, group_sources) in enumerate(grouped.values(), 1):
            cases_path = temporary_dir / f"{index:04d}.cases.json"
            _atomic_json(
                cases_path,
                [
                    {"id": source["id"], "image": source["file"], "split": source["split"]}
                    for source in group_sources
                ],
            )
            group_output = temporary_dir / f"dataset-{index:04d}"
            summaries.append(
                prepare_candidates(
                    cases_path,
                    batch_dir,
                    group_output,
                    config,
                    crop_backend=crop_backend,
                    roi_config_path=path,
                    teacher_model_dir=teacher_artifact[0] if teacher_artifact else None,
                    teacher_model_version=teacher_artifact[1] if teacher_artifact else None,
                    negative_examples_path=negative_path,
                )
            )
            if not output_dir.exists():
                shutil.copytree(group_output, output_dir)
            else:
                shutil.copytree(group_output / "images", output_dir / "images", dirs_exist_ok=True)
                _merge_crop_manifests(output_dir, group_output)

        merged_rows = {split: [] for split in ("train", "holdout")}
        for index in range(1, len(summaries) + 1):
            group_output = temporary_dir / f"dataset-{index:04d}"
            for split in merged_rows:
                path = group_output / "review" / f"{split}.jsonl"
                if path.is_file():
                    merged_rows[split].extend(
                        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                    )
        for split, rows in merged_rows.items():
            _atomic_jsonl(output_dir / "review" / f"{split}.jsonl", rows)

        result: dict[str, Any] = {}
        for summary in summaries:
            for key, value in summary.items():
                if isinstance(value, int):
                    result[key] = int(result.get(key, 0)) + value
                elif key not in result:
                    result[key] = value
        return result
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _review_row_match_score(previous: dict[str, Any], current: dict[str, Any]) -> float:
    if previous.get("source_id") != current.get("source_id") or previous.get("roi") != current.get("roi"):
        return 0.0

    previous_crop = previous.get("crop_sha256")
    current_crop = current.get("crop_sha256")
    if isinstance(previous_crop, str) and previous_crop and previous_crop == current_crop:
        return 2.0

    previous_texts = {
        canonicalize(str(previous.get(name)))
        for name in ("candidate_text", "rapidocr_text", "vision_text", "teacher_text", "transcription")
        if isinstance(previous.get(name), str) and str(previous.get(name)).strip()
    }
    current_texts = {
        canonicalize(str(current.get(name)))
        for name in ("candidate_text", "rapidocr_text", "vision_text", "teacher_text")
        if isinstance(current.get(name), str) and str(current.get(name)).strip()
    }
    if not previous_texts & current_texts:
        return 0.0

    previous_box = previous.get("global_box")
    current_box = current.get("global_box")
    if not isinstance(previous_box, list) or not isinstance(current_box, list):
        return 0.0
    if len(previous_box) != 4 or len(current_box) != 4:
        return 0.0
    try:
        first = tuple(float(value) for value in previous_box)
        second = tuple(float(value) for value in current_box)
    except (TypeError, ValueError):
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller_area = min(first_area, second_area)
    overlap = intersection / smaller_area if smaller_area else 0.0
    return 1.0 + overlap if overlap >= 0.75 else 0.0


def _inherit_manual_review(
    previous_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Carry only explicit human decisions to a newly generated candidate set."""
    manual_rows = [
        row
        for row in previous_rows
        if row.get("review_status") in {"accepted", "rejected"}
        and not row.get("auto_accept_reason")
        and not row.get("auto_reject_reason")
    ]
    used: set[int] = set()
    inherited = {"accepted": 0, "rejected": 0, "unmatched": 0}
    for row in current_rows:
        matches = sorted(
            (
                (score, index, previous)
                for index, previous in enumerate(manual_rows)
                if index not in used
                and (score := _review_row_match_score(previous, row)) > 0
            ),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if not matches or (len(matches) > 1 and matches[0][0] == matches[1][0]):
            continue
        _, index, previous = matches[0]
        used.add(index)
        row["review_status"] = str(previous["review_status"])
        row["transcription"] = previous.get("transcription") if row["review_status"] == "accepted" else None
        row["auto_accept_reason"] = None
        row["auto_reject_reason"] = None
        inherited[row["review_status"]] += 1

    inherited["unmatched"] = len(manual_rows) - len(used)
    return inherited


def recreate_candidates(
    batch_dir: Path,
    *,
    crop_backend: str = "rust",
) -> dict[str, Any]:
    """Rebuild candidates with current ROI rules while preserving safe human decisions."""
    dataset_dir = batch_dir / "dataset"
    previous_rows = {
        split: review_rows(batch_dir, split)
        for split in ("train", "holdout")
    }
    if dataset_dir.exists() and not all(
        (dataset_dir / "review" / f"{split}.jsonl").is_file() for split in ("train", "holdout")
    ):
        raise RuntimeError("current candidate output is incomplete; finish or remove the incomplete output before rebuilding")

    manifest = load_manifest(batch_dir)
    _materialize_source_layouts(batch_dir, manifest)
    teacher_artifact = _candidate_artifact()
    negative_path = rebuild_negative_registry(batch_dir)
    revision_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    temporary_dir = batch_dir / f".recreate-{revision_id}"
    temporary_output = temporary_dir / "dataset"
    temporary_dir.mkdir()
    inherited = {"accepted": 0, "rejected": 0, "unmatched": 0}
    try:
        generated = _prepare_candidate_groups(
            batch_dir,
            manifest.get("sources", []),
            temporary_output,
            roi_config_path=DEFAULT_ROI_CONFIG,
            crop_backend=crop_backend,
            teacher_artifact=teacher_artifact,
            negative_path=negative_path,
        )
        for split in ("train", "holdout"):
            current_rows = review_rows(temporary_dir, split)
            counts = _inherit_manual_review(previous_rows[split], current_rows)
            for key, value in counts.items():
                inherited[key] += value
            _atomic_jsonl(temporary_output / "review" / f"{split}.jsonl", current_rows)

        _atomic_json(
            temporary_output / "revision.json",
            {
                "schema_version": 1,
                "revision_id": revision_id,
                "created_at": datetime.now(UTC).isoformat(),
                "mode": "recreate",
                "inherited_manual_decisions": inherited,
            },
        )

        archived_dataset: str | None = None
        archive_dir = batch_dir / "dataset-revisions" / revision_id / "dataset"
        moved_previous = False
        try:
            if dataset_dir.exists():
                archive_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dataset_dir), str(archive_dir))
                moved_previous = True
                archived_dataset = str(archive_dir.relative_to(batch_dir))
            shutil.move(str(temporary_output), str(dataset_dir))
        except Exception:
            if moved_previous and not dataset_dir.exists():
                shutil.move(str(archive_dir), str(dataset_dir))
            raise

        manifest["active_dataset_revision"] = revision_id
        revisions = list(manifest.get("dataset_revisions", []))
        revisions.append(
            {
                "revision_id": revision_id,
                "created_at": datetime.now(UTC).isoformat(),
                "archived_previous_dataset": archived_dataset,
                "inherited_manual_decisions": inherited,
            }
        )
        manifest["dataset_revisions"] = revisions
        _atomic_json(batch_dir / "batch.json", manifest)
        return {
            **generated,
            "recreated_candidates": True,
            "revision_id": revision_id,
            "inherited_manual_accepted": inherited["accepted"],
            "inherited_manual_rejected": inherited["rejected"],
            "unmatched_manual_decisions": inherited["unmatched"],
            "reused_existing_candidates": False,
        }
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def generate_candidates(
    batch_dir: Path,
    *,
    roi_config_path: Path = DEFAULT_ROI_CONFIG,
    crop_backend: str = "rust",
) -> dict[str, Any]:
    dataset_dir = batch_dir / "dataset"
    teacher_artifact = _candidate_artifact()
    negative_path = rebuild_negative_registry(batch_dir)
    review_dir = dataset_dir / "review"
    review_files = [review_dir / "train.jsonl", review_dir / "holdout.jsonl"]
    if dataset_dir.exists():
        if not all(path.is_file() for path in review_files):
            raise RuntimeError(
                f"candidate output is incomplete: {dataset_dir}; create a new batch instead of overwriting private review data"
            )
        rows = {split: review_rows(batch_dir, split) for split in ("train", "holdout")}
        deduplicated = deduplicate_review_rows(batch_dir)
        if deduplicated:
            rows = {split: review_rows(batch_dir, split) for split in ("train", "holdout")}
        negative_auto_rejected = apply_negative_matches(batch_dir, negative_path)
        if negative_auto_rejected:
            rows = {split: review_rows(batch_dir, split) for split in ("train", "holdout")}
            _invalidate_labels(dataset_dir)
        known_sources = {str(row["source_id"]) for split_rows in rows.values() for row in split_rows}
        missing_sources = [source for source in load_manifest(batch_dir)["sources"] if str(source["id"]) not in known_sources]
        if missing_sources:
            temporary_dir = batch_dir / f".candidate-append-{uuid.uuid4().hex}"
            temporary_cases = temporary_dir / "cases.json"
            temporary_output = temporary_dir / "dataset"
            temporary_dir.mkdir()
            try:
                _atomic_json(
                    temporary_cases,
                    [{"id": row["id"], "image": row["file"], "split": row["split"]} for row in missing_sources],
                )
                prepared = _prepare_candidate_groups(
                    batch_dir,
                    missing_sources,
                    temporary_output,
                    roi_config_path=roi_config_path,
                    crop_backend=crop_backend,
                    teacher_artifact=teacher_artifact,
                    negative_path=negative_path,
                )
                for split in ("train", "holdout"):
                    rows[split].extend(review_rows(temporary_output, split))
                    _atomic_jsonl(review_dir / f"{split}.jsonl", rows[split])
                shutil.copytree(temporary_output / "images", dataset_dir / "images", dirs_exist_ok=True)
                if crop_backend == "rust":
                    _merge_crop_manifests(dataset_dir, temporary_output)
                _invalidate_labels(dataset_dir)
                deduplicated += int(prepared.get("deduplicated", 0))
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
            return {
                "cases": len(load_manifest(batch_dir)["sources"]),
                "train_cases": len({str(row["source_id"]) for row in rows["train"]}),
                "holdout_cases": len({str(row["source_id"]) for row in rows["holdout"]}),
                "train_candidates": len(rows["train"]),
                "holdout_candidates": len(rows["holdout"]),
                "auto_accepted": sum(row.get("auto_accept_reason") == "rapidocr_vision_agreement" for split_rows in rows.values() for row in split_rows),
                "auto_rejected": sum(bool(row.get("auto_reject_reason")) for split_rows in rows.values() for row in split_rows),
                "negative_auto_rejected": negative_auto_rejected,
                "deduplicated": deduplicated,
                "reused_existing_candidates": False,
            }
        return {
            "cases": len(load_manifest(batch_dir)["sources"]),
            "train_cases": len({str(row["source_id"]) for row in rows["train"]}),
            "holdout_cases": len({str(row["source_id"]) for row in rows["holdout"]}),
            "train_candidates": len(rows["train"]),
            "holdout_candidates": len(rows["holdout"]),
            "auto_accepted": sum(row.get("auto_accept_reason") == "rapidocr_vision_agreement" for split_rows in rows.values() for row in split_rows),
            "auto_rejected": sum(bool(row.get("auto_reject_reason")) for split_rows in rows.values() for row in split_rows),
            "negative_auto_rejected": negative_auto_rejected,
            "deduplicated": deduplicated,
            "reused_existing_candidates": True,
        }
    sources = load_manifest(batch_dir).get("sources", []) if (batch_dir / "batch.json").is_file() else []
    return _prepare_candidate_groups(
        batch_dir,
        sources,
        dataset_dir,
        roi_config_path=roi_config_path,
        crop_backend=crop_backend,
        teacher_artifact=teacher_artifact,
        negative_path=negative_path,
    )


def refresh_vision_candidates(
    batch_dir: Path,
    *,
    vision_factory: Callable[[], Any] = VisionOcr,
) -> dict[str, int]:
    """Refresh Vision fields while preserving all existing human review data."""
    dataset_dir = batch_dir / "dataset"
    review_dir = dataset_dir / "review"
    review_paths = {split: review_dir / f"{split}.jsonl" for split in ("train", "holdout")}
    if not all(path.is_file() for path in review_paths.values()):
        raise ValueError("generate candidates before refreshing Vision results")

    vision = vision_factory()
    updated: dict[str, list[dict[str, Any]]] = {}
    summary = {
        "rows": 0,
        "vision_covered": 0,
        "auto_accepted": 0,
        "auto_rejected": 0,
        "preserved_accepted": 0,
        "preserved_rejected": 0,
    }
    for split, path in review_paths.items():
        rows = review_rows(batch_dir, split)
        refreshed: list[dict[str, Any]] = []
        for row in rows:
            crop = Path(str(row.get("crop", "")))
            if crop.is_absolute() or ".." in crop.parts:
                raise ValueError(f"candidate contains an unsafe crop path: {crop}")
            crop_path = (dataset_dir / crop).resolve()
            if dataset_dir.resolve() not in crop_path.parents or not crop_path.is_file():
                raise ValueError(f"candidate crop does not exist: {crop}")

            lines = vision.recognize(decode_image(crop_path.read_bytes()))
            best: VisionLine | None = max(lines, key=lambda line: (line.confidence, len(line.text)), default=None)
            row["vision_text"] = best.text if best else None
            row["vision_confidence"] = round(best.confidence, 4) if best else None
            rapid_text = row.get("rapidocr_text")
            rapid_confidence = row.get("rapidocr_confidence")
            teacher_text = row.get("teacher_text")
            auto_reject_reason = candidate_rejection_reason(
                str(row.get("roi", "")),
                (rapid_text, best.text if best else None, teacher_text),
            )
            was_auto_rejected = bool(row.get("auto_reject_reason"))
            confidences = [value for value in (rapid_confidence, row["vision_confidence"]) if isinstance(value, (int, float))]
            if confidences:
                row["confidence"] = round(max(confidences), 4)
                row["candidate_confidence"] = row["confidence"]

            was_auto_accepted = row.get("auto_accept_reason") in {
                "rapidocr_vision_agreement",
                "rapidocr_vision_teacher_agreement",
            }
            agrees = engine_results_agree(
                (rapid_text if isinstance(rapid_text, str) else None, rapid_confidence if isinstance(rapid_confidence, (int, float)) else None),
                (best.text if best else None, best.confidence if best else None),
                (teacher_text if isinstance(teacher_text, str) else None, row.get("teacher_confidence") if isinstance(row.get("teacher_confidence"), (int, float)) else None),
                minimum_confidence=0.98,
            )
            if auto_reject_reason and (row.get("review_status") == "pending" or row.get("auto_accept_reason")):
                row["review_status"] = "rejected"
                row["review_method"] = "automatic"
                row["transcription"] = None
                row["auto_accept_reason"] = None
                row["auto_reject_reason"] = auto_reject_reason
            elif was_auto_rejected and not auto_reject_reason:
                row["review_status"] = "pending"
                row["transcription"] = None
                row["auto_reject_reason"] = None
            elif row.get("review_status") == "pending":
                if agrees:
                    row["review_status"] = "accepted"
                    row["review_method"] = "automatic"
                    row["transcription"] = canonicalize(rapid_text)
                    row["auto_accept_reason"] = "rapidocr_vision_teacher_agreement" if teacher_text else "rapidocr_vision_agreement"
            elif was_auto_accepted:
                if agrees:
                    row["auto_accept_reason"] = "rapidocr_vision_teacher_agreement" if teacher_text else "rapidocr_vision_agreement"
                else:
                    row["review_status"] = "pending"
                    row["transcription"] = None
                    row["auto_accept_reason"] = None

            teacher_confidence = row.get("teacher_confidence")
            teacher_rapid_agrees = engine_results_agree(
                (rapid_text if isinstance(rapid_text, str) else None, rapid_confidence if isinstance(rapid_confidence, (int, float)) else None),
                (row.get("vision_text") if isinstance(row.get("vision_text"), str) else None, row.get("vision_confidence") if isinstance(row.get("vision_confidence"), (int, float)) else None),
                (teacher_text if isinstance(teacher_text, str) else None, teacher_confidence if isinstance(teacher_confidence, (int, float)) else None),
                minimum_confidence=0.98,
            )
            if split == "train" and row.get("review_status") == "pending" and teacher_rapid_agrees:
                row["review_status"] = "accepted"
                row["review_method"] = "automatic"
                row["transcription"] = canonicalize(rapid_text)
                row["auto_accept_reason"] = "teacher_rapidocr_agreement"
            row["teacher_auto_accept_eligible"] = (
                split == "train"
                and row.get("review_status") == "pending"
                and teacher_rapid_agrees
            )
            row["teacher_suggestion"] = (
                isinstance(teacher_text, str)
                and bool(teacher_text.strip())
                and isinstance(teacher_confidence, (int, float))
                and teacher_confidence >= 0.95
                and (not row.get("candidate_text") or canonicalize(str(row["candidate_text"])) == canonicalize(teacher_text))
            )
            row["suggested_transcription"] = canonicalize(teacher_text) if row["teacher_suggestion"] else None

            summary["rows"] += 1
            if best is not None and best.text.strip():
                summary["vision_covered"] += 1
            if row.get("auto_accept_reason") in {"rapidocr_vision_agreement", "rapidocr_vision_teacher_agreement"}:
                summary["auto_accepted"] += 1
            if row.get("auto_reject_reason"):
                summary["auto_rejected"] += 1
            if row.get("review_status") == "accepted" and not row.get("auto_accept_reason"):
                summary["preserved_accepted"] += 1
            if row.get("review_status") == "rejected" and not row.get("auto_reject_reason"):
                summary["preserved_rejected"] += 1
            refreshed.append(row)
        updated[split] = refreshed

    for split, path in review_paths.items():
        _atomic_jsonl(path, updated[split])
    return summary


def refresh_teacher_candidates(
    batch_dir: Path,
    *,
    teacher_factory: Callable[[], Any] | None = None,
    teacher_model_dir: Path | None = None,
    teacher_model_version: str | None = None,
) -> dict[str, int | str | None]:
    """Add teacher predictions to existing crops without changing human decisions."""
    dataset_dir = batch_dir / "dataset"
    review_dir = dataset_dir / "review"
    review_paths = {split: review_dir / f"{split}.jsonl" for split in ("train", "holdout")}
    if not all(path.is_file() for path in review_paths.values()):
        raise ValueError("generate candidates before refreshing teacher results")
    if teacher_factory is not None:
        teacher = teacher_factory()
    else:
        artifact = (teacher_model_dir, teacher_model_version) if teacher_model_dir else _candidate_artifact()
        if artifact is None:
            raise ValueError("no complete local teacher artifact is available")
        teacher_model_dir, teacher_model_version = artifact
        teacher = create_artifact_ocr(teacher_model_dir)

    updated: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, int | str | None] = {
        "rows": 0,
        "teacher_covered": 0,
        "teacher_suggestions": 0,
        "teacher_auto_accept_eligible": 0,
        "teacher_auto_accepted": 0,
        "auto_rejected": 0,
        "preserved_accepted": 0,
        "preserved_rejected": 0,
        "teacher_model_version": teacher_model_version,
    }
    for split, path in review_paths.items():
        rows = review_rows(batch_dir, split)
        refreshed: list[dict[str, Any]] = []
        for row in rows:
            was_accepted = row.get("review_status") == "accepted"
            was_rejected = row.get("review_status") == "rejected" and not row.get("auto_reject_reason")
            crop = Path(str(row.get("crop", "")))
            if crop.is_absolute() or ".." in crop.parts:
                raise ValueError(f"candidate contains an unsafe crop path: {crop}")
            crop_path = (dataset_dir / crop).resolve()
            if dataset_dir.resolve() not in crop_path.parents or not crop_path.is_file():
                raise ValueError(f"candidate crop does not exist: {crop}")
            result = teacher(decode_image(crop_path.read_bytes()), use_det=False, use_cls=False)
            teacher_text, teacher_confidence = best_rapid_candidate(result)
            was_auto_accepted = bool(row.get("auto_accept_reason"))
            row["teacher_model_version"] = teacher_model_version
            row["teacher_text"] = teacher_text
            row["teacher_confidence"] = round(teacher_confidence, 4) if teacher_confidence is not None else None
            candidate_text = row.get("candidate_text")
            rapid_text = row.get("rapidocr_text")
            rapid_confidence = row.get("rapidocr_confidence")
            auto_reject_reason = candidate_rejection_reason(
                str(row.get("roi", "")),
                (rapid_text, row.get("vision_text"), teacher_text),
            )
            was_auto_rejected = bool(row.get("auto_reject_reason"))
            if auto_reject_reason and (row.get("review_status") == "pending" or row.get("auto_accept_reason")):
                row["review_status"] = "rejected"
                row["review_method"] = "automatic"
                row["transcription"] = None
                row["auto_accept_reason"] = None
                row["auto_reject_reason"] = auto_reject_reason
            elif was_auto_rejected and not auto_reject_reason:
                row["review_status"] = "pending"
                row["transcription"] = None
                row["auto_reject_reason"] = None
            teacher_rapid_agrees = engine_results_agree(
                (rapid_text if isinstance(rapid_text, str) else None, rapid_confidence if isinstance(rapid_confidence, (int, float)) else None),
                (row.get("vision_text") if isinstance(row.get("vision_text"), str) else None, row.get("vision_confidence") if isinstance(row.get("vision_confidence"), (int, float)) else None),
                (teacher_text if isinstance(teacher_text, str) else None, teacher_confidence if isinstance(teacher_confidence, (int, float)) else None),
                minimum_confidence=0.98,
            )
            if was_auto_accepted and not auto_reject_reason:
                if teacher_rapid_agrees:
                    if row.get("vision_text"):
                        row["auto_accept_reason"] = "rapidocr_vision_teacher_agreement"
                else:
                    row["review_status"] = "pending"
                    row["transcription"] = None
                    row["auto_accept_reason"] = None
            if split == "train" and row.get("review_status") == "pending" and teacher_rapid_agrees:
                row["review_status"] = "accepted"
                row["review_method"] = "automatic"
                row["transcription"] = canonicalize(rapid_text)
                row["auto_accept_reason"] = "teacher_rapidocr_agreement"
                summary["teacher_auto_accepted"] = int(summary["teacher_auto_accepted"]) + 1
            row["teacher_suggestion"] = (
                teacher_text is not None
                and teacher_confidence is not None
                and teacher_confidence >= 0.95
                and (not isinstance(candidate_text, str) or not candidate_text or canonicalize(candidate_text) == canonicalize(teacher_text))
            )
            row["suggested_transcription"] = canonicalize(teacher_text) if row["teacher_suggestion"] else None
            row["teacher_auto_accept_eligible"] = (
                split == "train"
                and row.get("review_status") == "pending"
                and teacher_rapid_agrees
            )
            summary["rows"] = int(summary["rows"]) + 1
            if teacher_text:
                summary["teacher_covered"] = int(summary["teacher_covered"]) + 1
            if row["teacher_suggestion"]:
                summary["teacher_suggestions"] = int(summary["teacher_suggestions"]) + 1
            if row["teacher_auto_accept_eligible"]:
                summary["teacher_auto_accept_eligible"] = int(summary["teacher_auto_accept_eligible"]) + 1
            if row.get("auto_reject_reason"):
                summary["auto_rejected"] = int(summary["auto_rejected"]) + 1
            if was_accepted:
                summary["preserved_accepted"] = int(summary["preserved_accepted"]) + 1
            if was_rejected:
                summary["preserved_rejected"] = int(summary["preserved_rejected"]) + 1
            refreshed.append(row)
        updated[split] = refreshed

    for split, path in review_paths.items():
        _atomic_jsonl(path, updated[split])
    return summary


def roi_preview_paths(batch_dir: Path) -> list[tuple[str, str]]:
    manifest = load_manifest(batch_dir)
    if not manifest["sources"]:
        return []
    source = batch_dir / manifest["sources"][0]["file"]
    image = decode_image(source.read_bytes())
    options = _roi_config_options(DEFAULT_ROI_CONFIG)
    _, roi_config = _source_roi_config(batch_dir, manifest["sources"][0], options)
    normalized, rois = crop_all_rois(image, roi_config)
    preview_dir = batch_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    normalized_path = preview_dir / "normalized.png"
    cv2.imwrite(str(normalized_path), normalized)
    previews: list[tuple[str, str]] = [(str(normalized_path), "normalized canvas")]
    for name, roi in rois.items():
        path = preview_dir / f"{name}.png"
        cv2.imwrite(str(path), roi)
        previews.append((str(path), name))
    return previews


def review_rows(batch_dir: Path, split: str, status: str = "all") -> list[dict[str, Any]]:
    path = batch_dir / "dataset/review" / f"{split}.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if status == "auto_accepted":
        rows = [row for row in rows if row.get("auto_accept_reason") == "rapidocr_vision_agreement"]
    elif status == "teacher_eligible":
        rows = [row for row in rows if row.get("teacher_auto_accept_eligible") is True and row.get("review_status") == "pending"]
    elif status != "all":
        rows = [row for row in rows if row.get("review_status") == status]
    return rows


def update_review_row(batch_dir: Path, split: str, crop: str, status: str, transcription: str | None) -> dict[str, Any]:
    if status not in {"accepted", "rejected"}:
        raise ValueError("review status must be accepted or rejected")
    path = batch_dir / "dataset/review" / f"{split}.jsonl"
    rows = review_rows(batch_dir, split)
    for row in rows:
        if row.get("crop") != crop:
            continue
        if status == "accepted" and not (transcription or "").strip():
            raise ValueError("accepted candidates require a transcription")
        previous_transcription = row.get("transcription")
        was_auto_accepted = bool(row.get("auto_accept_reason"))
        row["review_status"] = status
        row["transcription"] = transcription.strip() if status == "accepted" and transcription else None
        row["auto_accept_reason"] = (
            row.get("auto_accept_reason")
            if status == "accepted" and was_auto_accepted and row["transcription"] == previous_transcription
            else None
        )
        row["auto_reject_reason"] = None
        row["review_method"] = "automatic" if status == "accepted" and was_auto_accepted and row["transcription"] == previous_transcription else "human"
        _atomic_jsonl(path, rows)
        rebuild_negative_registry(batch_dir)
        return row
    raise ValueError("review candidate no longer exists")


def accept_teacher_suggestions(batch_dir: Path) -> dict[str, int]:
    """Accept only explicit high-confidence teacher suggestions in train data."""
    path = batch_dir / "dataset/review/train.jsonl"
    rows = review_rows(batch_dir, "train")
    accepted = 0
    for row in rows:
        transcription = row.get("teacher_text")
        if (
            row.get("review_status") == "pending"
            and row.get("teacher_auto_accept_eligible") is True
            and isinstance(transcription, str)
            and transcription.strip()
        ):
            row["review_status"] = "accepted"
            row["transcription"] = canonicalize(transcription)
            row["auto_accept_reason"] = "teacher_model_agreement"
            row["review_method"] = "automatic"
            accepted += 1
    _atomic_jsonl(path, rows)
    counts = review_counts(batch_dir)
    return {"accepted": accepted, "pending": counts["pending"], "teacher_eligible": counts["teacher_eligible"]}


def review_counts(batch_dir: Path) -> dict[str, int]:
    rows = review_rows(batch_dir, "train") + review_rows(batch_dir, "holdout")
    counts = Counter(str(row.get("review_status", "pending")) for row in rows)
    return {
        "total": len(rows),
        "accepted": counts["accepted"],
        "pending": counts["pending"],
        "rejected": counts["rejected"],
        "teacher_eligible": sum(row.get("teacher_auto_accept_eligible") is True and row.get("review_status") == "pending" for row in rows),
    }


def _validate_review_readiness(batch_dir: Path) -> None:
    manifest = load_manifest(batch_dir)
    sources = manifest["sources"]
    split_sources = Counter(str(source["split"]) for source in sources)
    if split_sources["train"] == 0 or split_sources["holdout"] == 0:
        raise ValueError(
            "training labels require at least two distinct source screenshots so one can be held out for evaluation; "
            "create a new batch with at least two unique images"
        )

    for split in ("train", "holdout"):
        rows = review_rows(batch_dir, split)
        pending = sum(row.get("review_status") == "pending" for row in rows)
        accepted = sum(row.get("review_status") == "accepted" for row in rows)
        if pending:
            raise ValueError(f"{split} review still has {pending} pending candidate(s); accept or reject every candidate first")
        if not accepted:
            raise ValueError(f"{split} requires at least one accepted transcription for training and evaluation")


def finalize_dataset(batch_dir: Path) -> dict[str, int]:
    _validate_review_readiness(batch_dir)
    dataset_dir = batch_dir / "dataset"
    result = finalize(dataset_dir)
    result["validated_train"] = validate_rec(dataset_dir / "labels/train.txt")
    result["validated_holdout"] = validate_rec(dataset_dir / "labels/holdout.txt")
    return result


def export_dataset(batch_dir: Path, *, destination_root: Path = PRIVATE_DATASET_ROOT) -> dict[str, Any]:
    """Copy a finalized batch into the private datasets submodule as an immutable package."""
    result = finalize_dataset(batch_dir)
    if destination_root == PRIVATE_DATASET_ROOT and not (ROOT / "datasets/.git").exists():
        raise ValueError("private datasets submodule is not initialized; run `git submodule update --init --recursive`")
    batch_id = str(load_manifest(batch_dir)["batch_id"])
    destination = destination_root / batch_id
    if destination.exists():
        raise ValueError(f"dataset export already exists: {destination}; exports are immutable")
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = destination_root / f".{batch_id}-{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(batch_dir / "dataset", temporary / "dataset")
        shutil.copy2(batch_dir / "batch.json", temporary / "batch.json")
        _atomic_json(
            temporary / "export.json",
            {"batch_id": batch_id, "exported_at": datetime.now(UTC).isoformat(), **result},
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"export_dir": str(destination), **result}
