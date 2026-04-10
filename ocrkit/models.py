from __future__ import annotations

from pydantic import BaseModel, ConfigDict, HttpUrl


def to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ExtractRequest(CamelModel):
    image_url: HttpUrl


class OcrTexts(CamelModel):
    top_bar: str
    center_banner: str
    left_panel: str


class Extracted(CamelModel):
    passed: bool
    player_name: str | None
    time_sec: float | None
    deaths: int | None
    skips: int | None
    map_label: str | None
    difficulty: str | None
    ocr_texts: OcrTexts


class TitleDecision(CamelModel):
    awarded_keys: list[str]
    not_awarded_keys: list[str]
    not_evaluated_keys: list[str]
    reasons: list[str]
    confidence: float


class ExtractResponse(CamelModel):
    extracted: Extracted
    title_decision: TitleDecision


class PingResponse(CamelModel):
    ok: bool
    rules_version: str | None
    loaded_at_epoch: float | None
