from __future__ import annotations

from dataclasses import dataclass
import re

from .normalize import parse_int, parse_time_to_seconds


@dataclass
class LeftPanel:
    heroes_completed: int | None
    heroes_total: int | None
    challenge_completed: bool | None
    total_deaths: int | None
    total_skips: int | None
    clear_time: str | None
    clear_time_seconds: float | None


def parse_left_panel(text: str) -> LeftPanel:
    compact = text.replace("\n", " ")

    hero_match = re.search(r"英雄\s*[:：]\s*(\S+)\s*/\s*(\S+)", compact)
    heroes_completed = parse_int(hero_match.group(1)) if hero_match else None
    heroes_total = parse_int(hero_match.group(2)) if hero_match else None

    deaths, skips = _parse_total_deaths_skips(compact)

    time_match = re.search(r"(?:通关)?总计(?:时|耗时)\s*([0-9OoIlSB小时分秒:\s\.]+)", compact)
    clear_time = time_match.group(1).strip() if time_match else None

    return LeftPanel(
        heroes_completed=heroes_completed,
        heroes_total=heroes_total,
        challenge_completed=("挑战完成" in compact) if compact else None,
        total_deaths=deaths,
        total_skips=skips,
        clear_time=clear_time,
        clear_time_seconds=parse_time_to_seconds(clear_time or "") if clear_time else None,
    )


def _parse_total_deaths_skips(compact: str) -> tuple[int | None, int | None]:
    normalized = compact.translate(str.maketrans({"／": "/", "｜": "/", "|": "/", " ": ""}))

    anchor = re.search(r"总计(?:死亡|阵亡)(?:/跳过|跳过|过)?", normalized)
    if not anchor:
        return None, None

    nearby = normalized[anchor.end() : anchor.end() + 28]
    nearby = nearby.replace("次", "")
    nearby = nearby.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")

    m = re.search(r"([0-9]+)\s*/\s*([0-9]+)", nearby)
    if not m:
        return None, None

    return parse_int(m.group(1)), parse_int(m.group(2))
