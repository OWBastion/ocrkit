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
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


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


def generate_candidates(batch_dir: Path, *, roi_config_path: Path = DEFAULT_ROI_CONFIG) -> dict[str, int]:
    dataset_dir = batch_dir / "dataset"
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


def finalize_dataset(batch_dir: Path) -> dict[str, int]:
    dataset_dir = batch_dir / "dataset"
    result = finalize(dataset_dir)
    result["validated_train"] = validate_rec(dataset_dir / "labels/train.txt")
    result["validated_holdout"] = validate_rec(dataset_dir / "labels/holdout.txt")
    return result
