from __future__ import annotations

from dataclasses import dataclass
import re

from .normalize import normalize_player_name, parse_time_to_seconds


_CENTER_PATTERNS = [
    re.compile(
        r"(?:祝)?贺\s*(?P<player>.+?)\s*以\s*(?P<deaths>\d+)\s*次阵亡\s*[&＆]\s*(?P<skips>\d+)\s*次跳过\s*[·.]?\s*耗时\s*(?P<time>.+?)\s*通关"
    )
]
_PLAYER_PATTERN = re.compile(r"(?:祝|兄)?贺\s*(?P<player>.+?)\s*以")


@dataclass
class CenterSummary:
    completed: bool
    player: str | None
    deaths: int | None
    skips: int | None
    duration_text: str | None
    duration_seconds: float | None


def parse_center_summary(text: str) -> CenterSummary:
    cleaned = text.replace("|", " ")
    for pattern in _CENTER_PATTERNS:
        m = pattern.search(cleaned)
        if not m:
            continue
        duration_text = m.group("time").strip()
        return CenterSummary(
            completed=True,
            player=normalize_player_name(m.group("player")),
            deaths=int(m.group("deaths")),
            skips=int(m.group("skips")),
            duration_text=duration_text,
            duration_seconds=parse_time_to_seconds(duration_text),
        )

    player_match = _PLAYER_PATTERN.search(cleaned)
    return CenterSummary(
        completed=("通关" in cleaned),
        player=normalize_player_name(player_match.group("player")) if player_match else None,
        deaths=None,
        skips=None,
        duration_text=None,
        duration_seconds=None,
    )
