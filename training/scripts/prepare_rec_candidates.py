from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.roi_config import RoiConfig, load_roi_config
from app.image.loader import decode_image
from app.image.preprocess import preprocess_by_roi
from app.image.roi import crop_all_rois
from app.parser.run_code import looks_like_run_code_value, parse_run_code
from training.vision import VisionLine, VisionOcr


DEFAULT_FIXTURES = ROOT / "datasets/fixtures/challenge"
DEFAULT_OUTPUT = ROOT / "datasets/labeled/rec"
DEFAULT_ROI_CONFIG = ROOT / "configs/roi_1280x720.yaml"

HOLDOUT_IDS = frozenset(
    {
        "samoa_hell_01",
        "route_66_01",
        "numbani_01",
        "new_junk_city_in_progress_01",
        "circuit_royal_01",
        "hanamura_02",
        "anubis_temple_01",
        "lijiang_tower_01",
    }
)
AUTO_ACCEPT_CONFIDENCE = 0.98
TEACHER_SUGGESTION_CONFIDENCE = 0.95
TEACHER_AUTO_ACCEPT_CONFIDENCE = 0.98
MATCH_IOU = 0.5
REQUIRED_ARTIFACT_FILES = ("rapidocr.yaml", "det.onnx", "rec.onnx", "rec_dict.txt")


@dataclass(frozen=True)
class CandidateLine:
    text: str
    confidence: float
    box: np.ndarray


def candidate_artifact_version(artifact_dir: Path) -> str:
    """Return the immutable model version for a validated local artifact."""
    if not artifact_dir.is_dir():
        raise ValueError(f"candidate artifact directory does not exist: {artifact_dir}")
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).is_file()]
    if missing:
        raise ValueError(f"candidate artifact is incomplete: {', '.join(missing)}")
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate artifact manifest is invalid: {manifest_path}") from exc
        version = manifest.get("version")
        if manifest.get("schema_version") != 1 or manifest.get("model") != "pp-ocrv6-small" or not isinstance(version, str):
            raise ValueError(f"candidate artifact manifest is unsupported: {manifest_path}")
        return version
    return artifact_dir.name


def discover_candidate_artifact(artifact_root: Path) -> tuple[Path, str] | None:
    """Find the newest complete local release artifact, if one is available."""
    if not artifact_root.is_dir():
        return None
    candidates: list[tuple[Path, str]] = []
    for artifact_dir in artifact_root.iterdir():
        if not artifact_dir.is_dir() or not (artifact_dir / "manifest.json").is_file():
            continue
        try:
            candidates.append((artifact_dir, candidate_artifact_version(artifact_dir)))
        except ValueError:
            continue
    return max(candidates, key=lambda item: item[1]) if candidates else None


def create_artifact_ocr(artifact_dir: Path) -> Any:
    candidate_artifact_version(artifact_dir)
    return RapidOCR(config_path=str(artifact_dir / "rapidocr.yaml"), params={"Global.use_cls": False})


def best_rapid_candidate(result: Any) -> tuple[str | None, float | None]:
    texts = list(result.txts or [])
    scores = [float(score) for score in (result.scores or [])]
    if not texts or not scores:
        return None, None
    index = max(range(min(len(texts), len(scores))), key=lambda item: scores[item])
    text = str(texts[index]).strip()
    return (text or None), scores[index]


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def candidate_rejection_reason(roi_name: str, texts: Iterable[str | None]) -> str | None:
    """Reject OCR text that cannot belong to a field with a strict format."""
    if roi_name == "run_code_panel" and not any(
        isinstance(text, str) and (parse_run_code(text).status == "ok" or looks_like_run_code_value(text))
        for text in texts
    ):
        return "run_code.content_mismatch"
    return None


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases.json must contain a list")
    return data


def split_for_case(case_id: str) -> str:
    return "holdout" if case_id in HOLDOUT_IDS else "train"


def _crop_line(image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
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


def _box_bounds(box: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _box_bounds(first)
    bx1, by1, bx2, by2 = _box_bounds(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def _union_box(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = _box_bounds(first)
    ox1, oy1, ox2, oy2 = _box_bounds(second)
    return np.array(
        [[min(x1, ox1), min(y1, oy1)], [max(x2, ox2), min(y1, oy1)], [max(x2, ox2), max(y2, oy2)], [min(x1, ox1), max(y2, oy2)]],
        dtype=np.float32,
    )


def _rapid_lines(result: Any) -> list[CandidateLine]:
    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    scores = result.scores if result.scores is not None else []
    return [
        CandidateLine(str(text).strip(), float(score), np.asarray(box, dtype=np.float32))
        for box, text, score in zip(boxes, texts, scores, strict=True)
        if str(text).strip()
    ]


def _vision_lines(result: list[VisionLine]) -> list[CandidateLine]:
    return [CandidateLine(line.text, line.confidence, line.box) for line in result]


def _paired_lines(rapid: list[CandidateLine], vision: list[CandidateLine]) -> list[tuple[CandidateLine | None, CandidateLine | None]]:
    pairs: list[tuple[CandidateLine | None, CandidateLine | None]] = []
    unmatched = set(range(len(vision)))
    for rapid_line in rapid:
        index, overlap = max(
            ((index, _iou(rapid_line.box, vision[index].box)) for index in unmatched),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        if overlap >= MATCH_IOU:
            pairs.append((rapid_line, vision[index]))
            unmatched.remove(index)
        else:
            pairs.append((rapid_line, None))
    pairs.extend((None, vision[index]) for index in sorted(unmatched))
    return pairs


def _paired_candidate_lines(
    rapid: list[CandidateLine],
    vision: list[CandidateLine],
    teacher: list[CandidateLine],
) -> list[tuple[CandidateLine | None, CandidateLine | None, CandidateLine | None]]:
    pairs: list[tuple[CandidateLine | None, CandidateLine | None, CandidateLine | None]] = []
    unmatched = set(range(len(teacher)))
    for rapid_line, vision_line in _paired_lines(rapid, vision):
        base_line = rapid_line or vision_line
        teacher_index, overlap = max(
            ((index, _iou(base_line.box, teacher[index].box)) for index in unmatched) if base_line else (),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        teacher_line = teacher[teacher_index] if overlap >= MATCH_IOU else None
        if teacher_line is not None:
            unmatched.remove(teacher_index)
        pairs.append((rapid_line, vision_line, teacher_line))
    pairs.extend((None, None, teacher[index]) for index in sorted(unmatched))
    return pairs


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_label_scaffold(output_dir: Path) -> None:
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "holdout"):
        (labels_dir / f"{split}.txt").write_text("", encoding="utf-8")
    (labels_dir / "README.md").write_text(
        "Review review/train.jsonl and review/holdout.jsonl first. For each accepted row, "
        "append `<crop>\\t<transcription>` to the corresponding labels/*.txt file. "
        "Do not include pending or rejected candidates.\n",
        encoding="utf-8",
    )


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


def _run_rust_crop_batch(
    cases_path: Path,
    fixture_dir: Path,
    roi_config_path: Path,
    workspace: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    layout_manifest = workspace / "layout.manifest.json"
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
    crop_root = workspace / "rust-crops"
    command = [
        *_rust_cli_command(),
        "crop-batch",
        "--manifest",
        str(layout_manifest),
        "--cases",
        str(cases_path),
        "--input-root",
        str(fixture_dir),
        "--output-dir",
        str(crop_root),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown Rust crop error").strip()
        raise RuntimeError(f"Rust crop batch failed: {detail}") from exc

    manifest_path = crop_root / "crop_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crops: dict[tuple[str, str], dict[str, str]] = {}
    for source in manifest.get("sources", []):
        source_id = str(source["source_id"])
        for roi_name, artifact in source["rois"].items():
            relative = Path(str(artifact["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Rust crop manifest contains an unsafe path: {relative}")
            crops[(source_id, str(roi_name))] = {
                "path": str(crop_root / relative),
                "sha256": str(artifact["sha256"]),
            }
    return crops


def prepare_candidates(
    cases_path: Path,
    fixture_dir: Path,
    output_dir: Path,
    roi_config: RoiConfig,
    ocr_factory: Callable[[], Any] = RapidOCR,
    vision_factory: Callable[[], Any] = VisionOcr,
    crop_backend: str = "python",
    roi_config_path: Path = DEFAULT_ROI_CONFIG,
    teacher_model_dir: Path | None = None,
    teacher_model_version: str | None = None,
    teacher_ocr_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Create editable recognition-label candidates without changing fixture files."""
    cases = load_cases(cases_path)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}; remove it after review or choose another path")
    if crop_backend not in {"python", "rust"}:
        raise ValueError(f"unsupported crop backend: {crop_backend}")

    ocr = ocr_factory()
    vision = vision_factory()
    teacher = teacher_ocr_factory() if teacher_ocr_factory is not None else (create_artifact_ocr(teacher_model_dir) if teacher_model_dir else None)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    rust_workspace: Path | None = None
    output_dir.mkdir(parents=True)
    try:
        rust_crops: dict[tuple[str, str], dict[str, str]] = {}
        if crop_backend == "rust":
            rust_workspace = Path(tempfile.mkdtemp(prefix=".rust-crops-", dir=str(output_dir.parent)))
            rust_crops = _run_rust_crop_batch(cases_path, fixture_dir, roi_config_path, rust_workspace)
            shutil.copy2(
                rust_workspace / "rust-crops/crop_manifest.json",
                output_dir / "crop_manifest.json",
            )

        for case in cases:
            case_id = str(case["id"])
            image_path = fixture_dir / str(case["image"])
            if not image_path.is_file():
                raise FileNotFoundError(f"fixture image does not exist: {image_path}")
            requested_split = case.get("split")
            if requested_split is not None and requested_split not in {"train", "holdout"}:
                raise ValueError(f"case {case_id} has invalid split: {requested_split}")
            split = str(requested_split) if requested_split else split_for_case(case_id)
            if crop_backend == "rust":
                rois = {}
                for roi_name in roi_config.rois:
                    artifact = rust_crops.get((case_id, roi_name))
                    if artifact is None:
                        raise RuntimeError(f"Rust crop manifest is missing {case_id}/{roi_name}")
                    raw_roi = cv2.imread(artifact["path"], cv2.IMREAD_COLOR)
                    if raw_roi is None:
                        raise RuntimeError(f"cannot read Rust crop: {artifact['path']}")
                    rois[roi_name] = raw_roi
            else:
                image = decode_image(image_path.read_bytes())
                _, rois = crop_all_rois(image, roi_config)

            for roi_name, roi in rois.items():
                processed = preprocess_by_roi(roi_name, roi)
                rapid = _rapid_lines(ocr(processed, use_cls=False))
                vision_lines = _vision_lines(vision.recognize(processed))
                teacher_lines = _rapid_lines(teacher(processed, use_cls=False)) if teacher is not None else []
                paired_lines = _paired_candidate_lines(rapid, vision_lines, teacher_lines)
                for index, (rapid_line, vision_line, teacher_line) in enumerate(paired_lines):
                    if rapid_line is not None and vision_line is not None:
                        box = _union_box(rapid_line.box, vision_line.box)
                    else:
                        box = (rapid_line or vision_line or teacher_line).box
                    crop = _crop_line(processed, box)
                    if crop is None:
                        continue
                    if crop.ndim == 2:
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    relative_crop = Path("images") / split / case_id / roi_name / f"{index:03d}.png"
                    crop_path = output_dir / relative_crop
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(crop_path), crop):
                        raise RuntimeError(f"failed to write crop: {crop_path}")
                    rapid_text = rapid_line.text if rapid_line else None
                    vision_text = vision_line.text if vision_line else None
                    teacher_text = teacher_line.text if teacher_line else None
                    candidate_text = rapid_text or vision_text or teacher_text
                    auto_reject_reason = candidate_rejection_reason(
                        roi_name,
                        (rapid_text, vision_text, teacher_text),
                    )
                    teacher_rapid_agrees = (
                        teacher_line is not None
                        and rapid_line is not None
                        and rapid_line.confidence >= TEACHER_AUTO_ACCEPT_CONFIDENCE
                        and teacher_line.confidence >= TEACHER_AUTO_ACCEPT_CONFIDENCE
                        and canonicalize(rapid_line.text) == canonicalize(teacher_line.text)
                    )
                    teacher_suggestion = (
                        teacher_line is not None
                        and bool(teacher_text)
                        and teacher_line.confidence >= TEACHER_SUGGESTION_CONFIDENCE
                        and (not candidate_text or canonicalize(candidate_text) == canonicalize(teacher_text))
                    )
                    auto_accepted = (
                        rapid_line is not None
                        and vision_line is not None
                        and rapid_line.confidence >= AUTO_ACCEPT_CONFIDENCE
                        and vision_line.confidence >= AUTO_ACCEPT_CONFIDENCE
                        and canonicalize(rapid_line.text) == canonicalize(vision_line.text)
                    )
                    teacher_auto_accepted = split == "train" and not auto_accepted and teacher_rapid_agrees
                    auto_accept_reason = (
                        "rapidocr_vision_agreement"
                        if auto_accepted
                        else "teacher_rapidocr_agreement"
                        if teacher_auto_accepted
                        else None
                    )
                    rows[split].append(
                        {
                            "crop": relative_crop.as_posix(),
                            "source_id": case_id,
                            "split": split,
                            "roi": roi_name,
                            "box": np.asarray(box, dtype=float).round(2).tolist(),
                            "candidate_text": candidate_text,
                            "confidence": round(max(line.confidence for line in (rapid_line, vision_line, teacher_line) if line), 4),
                            "rapidocr_text": rapid_text,
                            "rapidocr_confidence": round(rapid_line.confidence, 4) if rapid_line else None,
                            "vision_text": vision_text,
                            "vision_confidence": round(vision_line.confidence, 4) if vision_line else None,
                            "teacher_model_version": teacher_model_version,
                            "teacher_text": teacher_text,
                            "teacher_confidence": round(teacher_line.confidence, 4) if teacher_line else None,
                            "teacher_suggestion": teacher_suggestion,
                            "suggested_transcription": canonicalize(teacher_text) if teacher_suggestion and teacher_text else None,
                            "teacher_auto_accept_eligible": False,
                            "crop_backend": crop_backend,
                            "raw_roi_sha256": rust_crops.get((case_id, roi_name), {}).get("sha256"),
                            "review_status": "rejected" if auto_reject_reason else "accepted" if auto_accepted or teacher_auto_accepted else "pending",
                            "transcription": canonicalize(rapid_line.text) if not auto_reject_reason and (auto_accepted or teacher_auto_accepted) else None,
                            "auto_accept_reason": None if auto_reject_reason else auto_accept_reason,
                            "auto_reject_reason": auto_reject_reason,
                        }
                    )

        review_dir = output_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for split, split_rows in rows.items():
            _write_jsonl(review_dir / f"{split}.jsonl", split_rows)
        _write_label_scaffold(output_dir)
    except Exception:
        shutil.rmtree(output_dir)
        raise
    finally:
        if rust_workspace is not None:
            shutil.rmtree(rust_workspace, ignore_errors=True)

    return {
        "cases": len(cases),
        "train_cases": sum((str(case.get("split")) if case.get("split") else split_for_case(str(case["id"]))) == "train" for case in cases),
        "holdout_cases": sum((str(case.get("split")) if case.get("split") else split_for_case(str(case["id"]))) == "holdout" for case in cases),
        "train_candidates": len(rows["train"]),
        "holdout_candidates": len(rows["holdout"]),
        "auto_accepted": sum(row["review_status"] == "accepted" for split_rows in rows.values() for row in split_rows),
        "auto_rejected": sum(bool(row.get("auto_reject_reason")) for split_rows in rows.values() for row in split_rows),
        "teacher_auto_accepted": sum(row.get("auto_accept_reason") == "teacher_rapidocr_agreement" for split_rows in rows.values() for row in split_rows),
        "teacher_model_version": teacher_model_version,
        "teacher_suggestions": sum(row["teacher_suggestion"] for split_rows in rows.values() for row in split_rows),
        "teacher_auto_accept_eligible": sum(row["teacher_auto_accept_eligible"] for row in rows["train"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RapidOCR-assisted PP-OCR rec review candidates from fixtures.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--crop-backend", choices=("python", "rust"), default="python")
    parser.add_argument("--teacher-artifact-dir", type=Path)
    args = parser.parse_args()
    teacher_artifact = args.teacher_artifact_dir
    teacher_version = candidate_artifact_version(teacher_artifact) if teacher_artifact else None
    if teacher_artifact is None:
        discovered = discover_candidate_artifact(ROOT / "training/.work/artifacts")
        if discovered:
            teacher_artifact, teacher_version = discovered
    print(
        json.dumps(
            prepare_candidates(
                args.fixtures / "cases.json",
                args.fixtures,
                args.output,
                load_roi_config(args.roi_config),
                crop_backend=args.crop_backend,
                roi_config_path=args.roi_config,
                teacher_model_dir=teacher_artifact,
                teacher_model_version=teacher_version,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
