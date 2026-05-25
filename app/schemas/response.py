from __future__ import annotations

from pydantic import BaseModel


class DebugPayload(BaseModel):
    normalized_size: tuple[int, int]
    roi_coordinates: dict[str, dict[str, int]]
    raw_text: dict[str, str]
    confidence: dict[str, float]


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
    ok: bool
    data: ChallengeData | None = None
    warnings: list[str] = []
    debug: DebugPayload | None = None
