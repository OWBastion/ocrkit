"""Platform screenshot-set contract (issue #26, platform side owbastion.com#255).

The importer consumes ONE finalized platform screenshot set. The platform
supplies immutable set membership — object key, checksum, layout, and an
optional screenshot-level accuracy mark — and OCRKit downloads the evidence
objects directly from R2 with a read-only key scoped to the screenshot
evidence prefix. A finalized set's membership is the platform's explicit
approval for its members to be used as OCR training sources.

Contract endpoint (private, versioned):

- ``GET {base}/v1/ocrkit/screenshot-sets/{version}`` -> :class:`ScreenshotSetMetadata`

Authentication is a bearer token supplied out of band (never persisted). Every
model forbids extra keys so QQ identity, player-account internals, Grant/mastery
state, risk signals, submission decisions, and evidence URLs cannot be smuggled
into an imported batch or its logs.

``accuracy`` is a sampling/review hint only ("accurate" | "inaccurate" | null):
it orders or prioritizes human review and must never become a transcription or
a training label.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class ScreenshotSetMember(BaseModel):
    """One source screenshot belonging to a finalized screenshot set."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    layout_version: str = Field(min_length=1)
    accuracy: Literal["accurate", "inaccurate"] | None = None

    @model_validator(mode="after")
    def _validate_source_id_is_path_safe(self) -> "ScreenshotSetMember":
        if not _SAFE_SOURCE_ID.match(self.source_id):
            raise ValueError(f"unsafe source_id: {self.source_id!r}")
        return self

    @property
    def normalized_sha256(self) -> str:
        return self.sha256.lower()


class ScreenshotSetMetadata(BaseModel):
    """Immutable finalized screenshot-set identity and member manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    set_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    finalized: bool
    finalized_at: str = Field(min_length=1)
    members: list[ScreenshotSetMember] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_membership(self) -> "ScreenshotSetMetadata":
        source_ids: set[str] = set()
        object_keys: set[str] = set()
        for member in self.members:
            if member.source_id in source_ids:
                raise ValueError(f"duplicate source_id in screenshot set: {member.source_id}")
            if member.object_key in object_keys:
                raise ValueError(f"duplicate object_key in screenshot set: {member.object_key}")
            source_ids.add(member.source_id)
            object_keys.add(member.object_key)
        return self
