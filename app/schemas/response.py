from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TerminologyTokenMatch(BaseModel):
    raw: str
    normalized: str
    status: str
    match_type: str | None = None
    rule_id: str | None = None
    confidence: float = 0.0


class TerminologyNormalization(BaseModel):
    scope_id: str | None = None
    rules_version: str
    decision: str
    raw_text: str
    normalized_text: str
    tokens: list[TerminologyTokenMatch] = Field(default_factory=list)


class DebugPayload(BaseModel):
    normalized_size: tuple[int, int]
    roi_coordinates: dict[str, dict[str, int]]
    raw_text: dict[str, str]
    confidence: dict[str, float]
    terminology_normalization: dict[str, TerminologyNormalization] = Field(default_factory=dict)


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


class EventNumber(BaseModel):
    text: str
    value: float
    unit: str | None = None


class EventData(BaseModel):
    """The single active event shown under the stats block, as read; the platform judges it against the game version."""

    name: str | None = None
    duration_seconds: int | None = None
    description: list[str] = Field(default_factory=list)
    numbers: list[EventNumber] = Field(default_factory=list)
    text: str = ""


class ChallengeData(BaseModel):
    challenge_completed: bool | None
    heroes_completed: int | None
    heroes_total: int | None
    viewer_player: str | None
    achievement_title: str | None
    achievement_titles: list[str]
    achievement_unlocked: bool | None
    achievement_panel_text: str | None
    deaths: int | None
    skips: int | None
    duration_text: str | None
    duration_seconds: float | None
    map_name: str | None
    map_variant: str | None
    difficulty: str | None
    version: str | None
    run_code: str | None = None
    event: EventData | None = None
    ai_mark_detected: bool | None = None
    mode: str | None = None
    restart_in_seconds: float | None = None
    uptime_seconds: float | None = None
    server_load: int | None = None


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
