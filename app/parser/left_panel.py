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

    ds_match = re.search(r"总计死亡/跳过\s*(\S+)\s*/\s*(\S+)", compact)
    deaths = parse_int(ds_match.group(1)) if ds_match else None
    skips = parse_int(ds_match.group(2)) if ds_match else None

    time_match = re.search(r"通关总计时\s*([0-9OoIlSB小时分秒:\s\.]+)", compact)
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
