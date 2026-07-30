from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

from app.core.roi_config import load_roi_config
from app.image.loader import decode_image
from app.image.quality import assess_input_quality
from app.image.roi import crop_all_rois
from training.scripts.finalize_rec_labels import finalize
from training.scripts.prepare_rec_candidates import prepare_candidates
from training.scripts.validate_annotations import validate_rec

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = ROOT / "training/.work/studio"
DEFAULT_ROI_CONFIG = ROOT / "configs/roi_1280x720.yaml"
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


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
) -> tuple[Path, dict[str, Any]]:
    """Copy valid uploads into a private, ignored batch workspace and assign source-level splits."""
    candidates = [Path(item) for item in upload_paths if item]
    if not candidates:
        raise ValueError("select at least one image")
    roi_config = load_roi_config(roi_config_path)
    prepared: list[tuple[Path, str, dict[str, Any]]] = []
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
        prepared.append((source, digest, assess_input_quality(image, roi_config.width, roi_config.height)))
        seen.add(digest)
    if not prepared:
        raise ValueError("no supported, decodable images were selected")

    batch_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    batch_dir = work_root / "batches" / batch_id
    sources_dir = batch_dir / "sources"
    sources_dir.mkdir(parents=True)
    splits = _split_sources([digest for _, digest, _ in prepared], holdout_ratio)
    sources: list[dict[str, Any]] = []
    for index, (source, digest, quality) in enumerate(prepared, 1):
        extension = source.suffix.lower()
        target_name = f"{index:04d}-{digest[:12]}{extension}"
        shutil.copy2(source, sources_dir / target_name)
        sources.append(
            {
                "id": f"source-{digest[:12]}",
                "file": f"sources/{target_name}",
                "sha256": digest,
                "split": splits[digest],
                "original_name": source.name,
                "quality": quality,
            }
        )
    manifest = {
        "schema_version": "1",
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "layout_version": roi_config.version,
        "roi_config": str(roi_config_path.relative_to(ROOT)),
        "holdout_ratio": holdout_ratio,
        "sources": sources,
    }
    _atomic_json(batch_dir / "batch.json", manifest)
    _atomic_json(
        batch_dir / "cases.json",
        [{"id": row["id"], "image": row["file"], "split": row["split"]} for row in sources],
    )
    return batch_dir, batch_summary(batch_dir)


def append_sources(batch_dir: Path, upload_paths: list[str | Path]) -> dict[str, Any]:
    """Add new source screenshots without changing existing source-level splits."""
    manifest = load_manifest(batch_dir)
    candidates = [Path(item) for item in upload_paths if item]
    if not candidates:
        raise ValueError("select at least one image")
    roi_config = load_roi_config(ROOT / str(manifest["roi_config"]))
    existing = {str(row["sha256"]) for row in manifest["sources"]}
    prepared: list[tuple[Path, str, dict[str, Any]]] = []
    for source in candidates:
        if source.suffix.lower() not in _IMAGE_SUFFIXES or not source.is_file():
            continue
        digest = _digest(source)
        if digest in existing:
            continue
        image = decode_image(source.read_bytes())
        prepared.append((source, digest, assess_input_quality(image, roi_config.width, roi_config.height)))
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
    for index, (source, digest, quality) in enumerate(sorted(prepared, key=lambda row: row[1])):
        extension = source.suffix.lower()
        target_name = f"{next_index + index:04d}-{digest[:12]}{extension}"
        shutil.copy2(source, sources_dir / target_name)
        added.append(
            {
                "id": f"source-{digest[:12]}",
                "file": f"sources/{target_name}",
                "sha256": digest,
                "split": "holdout" if index < new_holdout else "train",
                "original_name": source.name,
                "quality": quality,
            }
        )
    manifest["sources"] = [*sources, *added]
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
    }


def generate_candidates(batch_dir: Path, *, roi_config_path: Path = DEFAULT_ROI_CONFIG) -> dict[str, int | bool]:
    dataset_dir = batch_dir / "dataset"
    review_dir = dataset_dir / "review"
    review_files = [review_dir / "train.jsonl", review_dir / "holdout.jsonl"]
    if dataset_dir.exists():
        if not all(path.is_file() for path in review_files):
            raise RuntimeError(
                f"candidate output is incomplete: {dataset_dir}; create a new batch instead of overwriting private review data"
            )
        rows = {split: review_rows(batch_dir, split) for split in ("train", "holdout")}
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
                prepare_candidates(temporary_cases, batch_dir, temporary_output, load_roi_config(roi_config_path))
                for split in ("train", "holdout"):
                    rows[split].extend(review_rows(temporary_output, split))
                    _atomic_jsonl(review_dir / f"{split}.jsonl", rows[split])
                shutil.copytree(temporary_output / "images", dataset_dir / "images", dirs_exist_ok=True)
                _invalidate_labels(dataset_dir)
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
            return {
                "cases": len(load_manifest(batch_dir)["sources"]),
                "train_cases": len({str(row["source_id"]) for row in rows["train"]}),
                "holdout_cases": len({str(row["source_id"]) for row in rows["holdout"]}),
                "train_candidates": len(rows["train"]),
                "holdout_candidates": len(rows["holdout"]),
                "auto_accepted": sum(row.get("auto_accept_reason") == "rapidocr_vision_agreement" for split_rows in rows.values() for row in split_rows),
                "reused_existing_candidates": False,
            }
        return {
            "cases": len(load_manifest(batch_dir)["sources"]),
            "train_cases": len({str(row["source_id"]) for row in rows["train"]}),
            "holdout_cases": len({str(row["source_id"]) for row in rows["holdout"]}),
            "train_candidates": len(rows["train"]),
            "holdout_candidates": len(rows["holdout"]),
            "auto_accepted": sum(row.get("auto_accept_reason") == "rapidocr_vision_agreement" for split_rows in rows.values() for row in split_rows),
            "reused_existing_candidates": True,
        }
    return prepare_candidates(batch_dir / "cases.json", batch_dir, dataset_dir, load_roi_config(roi_config_path))


def roi_preview_paths(batch_dir: Path) -> list[tuple[str, str]]:
    manifest = load_manifest(batch_dir)
    if not manifest["sources"]:
        return []
    source = batch_dir / manifest["sources"][0]["file"]
    image = decode_image(source.read_bytes())
    normalized, rois = crop_all_rois(image, load_roi_config(DEFAULT_ROI_CONFIG))
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
    if status != "all":
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
        row["review_status"] = status
        row["transcription"] = transcription.strip() if status == "accepted" and transcription else None
        row["auto_accept_reason"] = row.get("auto_accept_reason") if status == "accepted" else None
        _atomic_jsonl(path, rows)
        return row
    raise ValueError("review candidate no longer exists")


def review_counts(batch_dir: Path) -> dict[str, int]:
    rows = review_rows(batch_dir, "train") + review_rows(batch_dir, "holdout")
    counts = Counter(str(row.get("review_status", "pending")) for row in rows)
    return {"total": len(rows), "accepted": counts["accepted"], "pending": counts["pending"], "rejected": counts["rejected"]}


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
