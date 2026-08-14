"""Replayable evaluation records for constrained text adjudication.

Each record is the minimum text metadata needed to evaluate an unresolved OCR
terminology case:

- schema version, layout version, ROI / field family;
- per-engine OCR text and confidence (RapidOCR, optional Vision, previous model);
- the deterministic normalization result from #3;
- the allowed canonical candidates (allowlist) and their terminology version;
- reviewed ground truth (exact visible transcription and canonical value);
- a stable input digest over everything except ground truth.

Records forbid extra keys so images, private object URLs, player identity, QQ
data, submission decisions, and unrelated production metadata cannot be smuggled
into an experiment or sent to a provider.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


class EngineCandidate(BaseModel):
    engine: str
    text: str | None = None
    confidence: float | None = None


class NormalizationResult(BaseModel):
    decision: Literal["normalized", "unchanged", "ambiguous", "unresolved"]
    rules_version: str | None = None
    scope_id: str | None = None
    normalized_text: str | None = None


class ReviewedAnnotationRecord(BaseModel):
    """One reviewed annotation plus the OCR evidence used to evaluate it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    record_id: str
    layout_version: str
    roi: str
    field_family: str | None = None
    engines: list[EngineCandidate] = Field(min_length=1)
    candidates: list[str] = Field(min_length=1)
    candidates_version: str
    normalization: NormalizationResult | None = None
    truth: str
    canonical_truth: str | None = None
    input_digest: str | None = None

    @model_validator(mode="after")
    def _verify_input_digest(self) -> ReviewedAnnotationRecord:
        if self.input_digest is not None and self.input_digest != compute_input_digest(self):
            raise ValueError(f"input_digest mismatch for record {self.record_id!r}")
        return self


def _digest_payload(record: ReviewedAnnotationRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "layout_version": record.layout_version,
        "roi": record.roi,
        "field_family": record.field_family,
        "engines": [
            {"engine": item.engine, "text": item.text, "confidence": item.confidence}
            for item in record.engines
        ],
        "candidates": list(record.candidates),
        "candidates_version": record.candidates_version,
        "normalization": record.normalization.model_dump() if record.normalization else None,
    }


def compute_input_digest(record: ReviewedAnnotationRecord) -> str:
    """Stable digest of the adjudication input (ground truth excluded)."""
    payload = json.dumps(_digest_payload(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def primary_engine_text(record: ReviewedAnnotationRecord) -> str | None:
    """The most confident non-empty engine text, used as the deterministic input."""
    available = [item for item in record.engines if item.text and item.text.strip()]
    if not available:
        return None
    return max(available, key=lambda item: item.confidence if item.confidence is not None else 0.0).text


def load_records(path) -> list[ReviewedAnnotationRecord]:
    records: list[ReviewedAnnotationRecord] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = ReviewedAnnotationRecord.model_validate_json(line)
            records.append(record)
    return records
