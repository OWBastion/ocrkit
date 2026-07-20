from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DebugPayload(BaseModel):
    normalized_size: tuple[int, int]
    roi_coordinates: dict[str, dict[str, int]]
    raw_text: dict[str, str]
    confidence: dict[str, float]


class FieldEvidence(BaseModel):
    value: Any = None
    confidence: float = 0.0
    source_roi: list[str] = Field(default_factory=list)
    normalization: list[str] = Field(default_factory=list)
    status: str = "missing"


class QualityPayload(BaseModel):
    original_size: tuple[int, int]
    aspect_ratio: float
    layout_confidence: float
    cropped: bool
    blur_score: float
    normalized_size: tuple[int, int]
    layout_version: str
    warnings: list[str] = Field(default_factory=list)


class ChallengeData(BaseModel):
    challenge_completed: bool | None
    heroes_completed: int | None
    heroes_total: int | None
    player: str | None
    deaths: int | None
    skips: int | None
    duration_text: str | None
    duration_seconds: float | None
    map_name: str | None
    difficulty: str | None
    version: str | None


class ChallengeResponse(BaseModel):
    schema_version: str = "1"
    request_id: str
    engine: str
    model_version: str
    layout_version: str
    ok: bool
    data: ChallengeData | None = None
    fields: dict[str, FieldEvidence] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    quality: QualityPayload
    debug: DebugPayload | None = None
