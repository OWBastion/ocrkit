"""Replayable pre-review records for the #25 Jev-Omni sidecar experiment.

Each record is one Studio review row materialized for the bounded decision
task: the crop identity, the de-duplicated OCR candidate texts plus the two
fixed routing options, the deterministic-rule outcome that already decided the
row (if any), and the human review truth used only for scoring.

Records forbid extra keys so unrelated batch metadata cannot silently enter
the experiment input or its digest. Ground truth is stored on the record for
evaluation but is excluded from the input digest and from everything sent to
the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPTION_NONE_CORRECT = "None of the options above matches the image text exactly"
OPTION_NOT_VALID = "The image does not contain a valid target text for this field"
FIXED_OPTIONS = (OPTION_NONE_CORRECT, OPTION_NOT_VALID)

PROMPT_VERSION = "jev-prereview-v2"


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


class EngineCandidate(BaseModel):
    engine: str
    text: str
    confidence: float | None = None


class PreReviewRecord(BaseModel):
    """One Studio review row as a bounded pre-review decision input."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    record_id: str
    batch_id: str
    source_id: str
    split: Literal["train", "holdout"]
    roi: str
    layout_version: str | None = None
    crop: str
    crop_sha256: str
    engines: list[EngineCandidate]
    options: list[str] = Field(min_length=2)
    prompt_version: str = PROMPT_VERSION
    deterministic: Literal["auto_accept", "auto_reject"] | None = None
    deterministic_reason: str | None = None
    truth_status: Literal["accepted", "rejected"] | None = None
    truth_transcription: str | None = None
    input_digest: str | None = None

    @property
    def residual(self) -> bool:
        """Rows the deterministic auto-accept/auto-reject rules did not decide."""
        return self.deterministic is None

    @property
    def candidate_options(self) -> list[str]:
        return self.options[: -len(FIXED_OPTIONS)]

    @model_validator(mode="after")
    def _verify(self) -> PreReviewRecord:
        if self.options[-len(FIXED_OPTIONS) :] != list(FIXED_OPTIONS):
            raise ValueError(f"record {self.record_id!r} options must end with the fixed routing options")
        canonical = [canonicalize(option) for option in self.candidate_options]
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"record {self.record_id!r} has duplicate candidate options")
        if self.truth_status == "accepted" and not self.truth_transcription:
            raise ValueError(f"record {self.record_id!r} is accepted without a transcription")
        if self.input_digest is not None and self.input_digest != compute_input_digest(self):
            raise ValueError(f"input_digest mismatch for record {self.record_id!r}")
        return self


def _digest_payload(record: PreReviewRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "roi": record.roi,
        "layout_version": record.layout_version,
        "crop_sha256": record.crop_sha256,
        "options": list(record.options),
        "prompt_version": record.prompt_version,
    }


def compute_input_digest(record: PreReviewRecord) -> str:
    """Stable digest of the decision input (truth and rule outcomes excluded)."""
    payload = json.dumps(_digest_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_options(row: dict[str, object]) -> tuple[list[EngineCandidate], list[str]]:
    """Collect engine texts and build the de-duplicated option list.

    Candidate order is canonical and deterministic: the displayed primary
    candidate first, then RapidOCR, Vision, and previous-model (teacher) text.
    The two fixed routing options always come last.
    """
    engines: list[EngineCandidate] = []
    for engine, text_key, confidence_key in (
        ("candidate", "candidate_text", "confidence"),
        ("rapidocr", "rapidocr_text", "rapidocr_confidence"),
        ("vision", "vision_text", "vision_confidence"),
        ("teacher", "teacher_text", "teacher_confidence"),
    ):
        text = row.get(text_key)
        if not isinstance(text, str) or not text.strip():
            continue
        confidence = row.get(confidence_key)
        engines.append(
            EngineCandidate(
                engine=engine,
                text=text,
                confidence=round(float(confidence), 4) if isinstance(confidence, (int, float)) else None,
            )
        )
    seen: set[str] = set()
    options: list[str] = []
    for engine in engines:
        key = canonicalize(engine.text)
        if key in seen:
            continue
        seen.add(key)
        options.append(engine.text)
    return engines, options + list(FIXED_OPTIONS)


def record_from_review_row(
    batch_id: str,
    split: str,
    layout_version: str | None,
    crop_sha256: str,
    row: dict[str, object],
) -> PreReviewRecord:
    crop = str(row.get("crop", ""))
    auto_accept = row.get("auto_accept_reason")
    auto_reject = row.get("auto_reject_reason")
    deterministic: Literal["auto_accept", "auto_reject"] | None = None
    deterministic_reason: str | None = None
    if isinstance(auto_accept, str) and auto_accept:
        deterministic, deterministic_reason = "auto_accept", auto_accept
    elif isinstance(auto_reject, str) and auto_reject:
        deterministic, deterministic_reason = "auto_reject", auto_reject
    status = row.get("review_status")
    truth_status = status if status in ("accepted", "rejected") else None
    transcription = row.get("transcription")
    engines, options = build_options(row)
    return PreReviewRecord(
        schema_version="1",
        record_id=f"{batch_id}:{split}:{crop}",
        batch_id=batch_id,
        source_id=str(row.get("source_id", "")),
        split=split,  # type: ignore[arg-type]
        roi=str(row.get("roi", "")),
        layout_version=row.get("layout_version") if isinstance(row.get("layout_version"), str) else layout_version,
        crop=crop,
        crop_sha256=crop_sha256,
        engines=engines,
        options=options,
        deterministic=deterministic,
        deterministic_reason=deterministic_reason,
        truth_status=truth_status,  # type: ignore[arg-type]
        truth_transcription=transcription if truth_status == "accepted" and isinstance(transcription, str) else None,
    )


def load_studio_records(batches_root: Path) -> tuple[list[PreReviewRecord], dict[str, Path]]:
    """Materialize pre-review records from Studio batch review files.

    Reads only the active ``dataset/review/*.jsonl`` state of each batch;
    archived ``dataset-revisions/`` are not replayed. Returns the records and a
    mapping from record_id to the absolute crop path for the runner.
    """
    records: list[PreReviewRecord] = []
    crop_paths: dict[str, Path] = {}
    missing: list[str] = []
    for batch_dir in sorted(path for path in batches_root.iterdir() if path.is_dir()):
        review_dir = batch_dir / "dataset" / "review"
        if not review_dir.is_dir():
            continue
        manifest_path = batch_dir / "batch.json"
        batch_layout = None
        if manifest_path.is_file():
            batch_layout = json.loads(manifest_path.read_text(encoding="utf-8")).get("layout_version")
        for split in ("train", "holdout"):
            review_path = review_dir / f"{split}.jsonl"
            if not review_path.is_file():
                continue
            for line in review_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = f"{batch_dir.name}:{split}:{row.get('crop', '')}"
                crop = Path(str(row.get("crop", "")))
                if crop.is_absolute() or ".." in crop.parts:
                    raise ValueError(f"review row {record_id!r} has an unsafe crop path")
                crop_path = (batch_dir / "dataset" / crop).resolve()
                if not crop_path.is_file():
                    missing.append(record_id)
                    continue
                record = record_from_review_row(
                    batch_dir.name, split, batch_layout, _sha256_file(crop_path), row
                )
                record.input_digest = compute_input_digest(record)
                records.append(record)
                crop_paths[record.record_id] = crop_path
    if missing:
        raise ValueError(f"review rows reference missing crops: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return records, crop_paths


def load_records(path: Path) -> list[PreReviewRecord]:
    records: list[PreReviewRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(PreReviewRecord.model_validate_json(line))
    return records


def write_records(path: Path, records: list[PreReviewRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
