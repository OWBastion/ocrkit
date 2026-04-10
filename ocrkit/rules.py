from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, time

import aiohttp

from .config import SETTINGS
from .models import Extracted, TitleDecision

CALCULABLE_KEYS = {
    "FLAWLESS",
    "DODGE_ULTIMATE",
    "SPEEDRUN",
    "CHALLENGER_LEGEND",
    "TRAVELER_HELL",
    "CONQUEROR",
    "DOMINATOR",
}
LEGENDARY_PLUS = {"传奇", "地狱"}


@dataclass
class TitleRules:
    version: str | None
    loaded_at_epoch: float
    active_title_keys: set[str]
    map_labels: list[str]


class TitleRulesStore:
    def __init__(self) -> None:
        self._rules: TitleRules | None = None
        self._expires_at = 0.0

    async def get_rules(self, session: aiohttp.ClientSession) -> TitleRules:
        now = monotonic()
        if self._rules and now < self._expires_at:
            return self._rules

        async with session.get(SETTINGS.title_source_url) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        titles = payload.get("titles", [])
        active_title_keys = {
            item.get("key")
            for item in titles
            if item.get("availability") == "active" and item.get("key")
        }
        map_labels = [item.get("mapLabel") for item in payload.get("mapTitles", []) if item.get("mapLabel")]
        version = payload.get("meta", {}).get("sourceLabel")

        self._rules = TitleRules(
            version=version,
            loaded_at_epoch=time(),
            active_title_keys=active_title_keys,
            map_labels=map_labels,
        )
        self._expires_at = now + SETTINGS.title_cache_ttl_sec
        return self._rules


def evaluate_titles(extracted: Extracted, rules: TitleRules) -> TitleDecision:
    awarded: list[str] = []
    not_awarded: list[str] = []
    not_evaluated: list[str] = []
    reasons: list[str] = []

    def mark(key: str, value: bool) -> None:
        (awarded if value else not_awarded).append(key)

    if extracted.passed:
        if "FLAWLESS" in rules.active_title_keys:
            if extracted.skips is None:
                not_evaluated.append("FLAWLESS")
                reasons.append("FLAWLESS: 缺少 skips")
            else:
                mark("FLAWLESS", extracted.skips == 0)

        if "DODGE_ULTIMATE" in rules.active_title_keys:
            if extracted.deaths is None:
                not_evaluated.append("DODGE_ULTIMATE")
                reasons.append("DODGE_ULTIMATE: 缺少 deaths")
            else:
                mark("DODGE_ULTIMATE", extracted.deaths <= 15)

        if "SPEEDRUN" in rules.active_title_keys:
            if extracted.time_sec is None or extracted.difficulty is None:
                not_evaluated.append("SPEEDRUN")
                reasons.append("SPEEDRUN: 缺少 time/difficulty")
            else:
                mark("SPEEDRUN", extracted.difficulty in LEGENDARY_PLUS and extracted.time_sec < 5400)

        if "CHALLENGER_LEGEND" in rules.active_title_keys:
            if extracted.difficulty is None:
                not_evaluated.append("CHALLENGER_LEGEND")
                reasons.append("CHALLENGER_LEGEND: 缺少 difficulty")
            else:
                mark("CHALLENGER_LEGEND", extracted.difficulty == "传奇")

        if "TRAVELER_HELL" in rules.active_title_keys:
            if extracted.difficulty is None:
                not_evaluated.append("TRAVELER_HELL")
                reasons.append("TRAVELER_HELL: 缺少 difficulty")
            else:
                mark("TRAVELER_HELL", extracted.difficulty == "地狱")

        if "CONQUEROR" in rules.active_title_keys:
            if extracted.map_label is None or extracted.difficulty is None:
                not_evaluated.append("CONQUEROR")
                reasons.append("CONQUEROR: 缺少 map/difficulty")
            else:
                mark("CONQUEROR", extracted.difficulty == "传奇")

        if "DOMINATOR" in rules.active_title_keys:
            if extracted.map_label is None or extracted.difficulty is None:
                not_evaluated.append("DOMINATOR")
                reasons.append("DOMINATOR: 缺少 map/difficulty")
            else:
                mark("DOMINATOR", extracted.difficulty == "地狱")
    else:
        for key in CALCULABLE_KEYS:
            if key in rules.active_title_keys:
                not_awarded.append(key)

    for key in sorted(rules.active_title_keys):
        if key not in CALCULABLE_KEYS:
            not_evaluated.append(key)

    total = len(awarded) + len(not_awarded) + len(not_evaluated)
    confidence = round((len(awarded) + len(not_awarded)) / total, 3) if total else 0.0

    return TitleDecision(
        awarded_keys=sorted(set(awarded)),
        not_awarded_keys=sorted(set(not_awarded)),
        not_evaluated_keys=sorted(set(not_evaluated)),
        reasons=reasons,
        confidence=confidence,
    )
