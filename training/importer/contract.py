"""Platform reviewed-snapshot contract (issue #5).

The importer consumes ONE finalized platform-reviewed dataset snapshot. The
platform supplies reviewed annotation facts and bounded evidence access; it
must never generate PaddleOCR labels or train/holdout splits, and OCRKit never
browses platform databases or buckets.

Contract endpoints (private, versioned):

- ``GET {base}/api/v1/snapshots/{snapshot_id}`` -> :class:`SnapshotMetadata`
- ``GET {base}/api/v1/snapshots/{snapshot_id}/annotations`` -> :class:`AnnotationsPayload`
- ``GET {base}/api/v1/objects/{object_id}/download`` -> image bytes (bounded access)

Authentication is a bearer token supplied out of band (never persisted). Every
model forbids extra keys so QQ identity, player-account internals, Grant/mastery
state, risk signals, submission decisions, image bytes, and object URLs cannot
be smuggled into a materialized import or its logs.

Annotation semantics stay distinct:

- ``ocr_prediction``: the original OCR output;
- ``exact_transcription``: the reviewed exact visible text;
- ``canonical_value``: the business-normalized platform value.

A reviewed annotation refers to one crop source, chosen by priority:

1. ``crop_object_id``: a platform pre-cropped training sample;
2. ``box``: a text-line polygon in standard-size coordinates (line crop derived
   locally from the normalized source);
3. ``roi``: a known field in a versioned layout (ROI crop derived with OCRKit's
   existing layout/ROI tooling).

A canonical business value is never converted into an OCR label; labels come
only from ``exact_transcription``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SnapshotObject(BaseModel):
    """One evidence object belonging to the snapshot (source or pre-crop)."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    kind: Literal["source", "crop"]
    sha256: str
    mime_type: str
    size_bytes: int
    source_id: str | None = None  # required for kind == "source"
    annotation_id: str | None = None  # required for kind == "crop"
    layout_version: str | None = None  # layout the source screenshot belongs to


class SnapshotMetadata(BaseModel):
    """Immutable finalized snapshot identity and member-object manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    snapshot_id: str
    version: str
    finalized: bool
    finalized_at: str | None = None
    objects: list[SnapshotObject] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_membership(self) -> SnapshotMetadata:
        seen: set[str] = set()
        for obj in self.objects:
            if obj.object_id in seen:
                raise ValueError(f"duplicate object_id in snapshot: {obj.object_id}")
            seen.add(obj.object_id)
            if obj.kind == "source" and not obj.source_id:
                raise ValueError(f"source object {obj.object_id} is missing source_id")
            if obj.kind == "crop" and not obj.annotation_id:
                raise ValueError(f"crop object {obj.object_id} is missing annotation_id")
        return self

    @property
    def sources(self) -> list[SnapshotObject]:
        return [obj for obj in self.objects if obj.kind == "source"]


class ReviewedAnnotation(BaseModel):
    """One reviewed annotation with distinct OCR / transcription / canonical values."""

    model_config = ConfigDict(extra="forbid")

    annotation_id: str
    source_id: str
    layout_version: str
    roi: str | None = None
    field: str | None = None
    ocr_prediction: str | None = None
    exact_transcription: str
    canonical_value: str | None = None
    box: list[list[float]] | None = None
    crop_object_id: str | None = None

    @model_validator(mode="after")
    def _validate_box_and_crop_reference(self) -> ReviewedAnnotation:
        if self.box is not None and (len(self.box) != 4 or any(len(point) != 2 for point in self.box)):
            raise ValueError(f"annotation {self.annotation_id} box must have four [x, y] points")
        if not self.exact_transcription.strip():
            raise ValueError(f"annotation {self.annotation_id} needs a non-empty exact_transcription")
        if self.box is not None and self.crop_object_id is not None:
            raise ValueError(f"annotation {self.annotation_id} must not declare both box and crop_object_id")
        if self.crop_object_id is None and self.box is None and not self.roi:
            raise ValueError(f"annotation {self.annotation_id} needs a crop source (crop_object_id, box, or roi)")
        return self


class AnnotationsPayload(BaseModel):
    """Reviewed annotation records for exactly one snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    snapshot_id: str
    annotations: list[ReviewedAnnotation] = Field(min_length=1)
