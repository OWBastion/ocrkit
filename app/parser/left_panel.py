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
    achievement_title: str | None = None
    achievement_titles: tuple[str, ...] = ()
    achievement_unlocked: bool | None = None
    achievement_challenge_disabled: bool | None = None


def parse_left_panel(text: str) -> LeftPanel:
    compact = text.replace("\n", " ")

    hero_match = re.search(r"英雄\s*[:：]\s*(\S+)\s*/\s*(\S+)", compact)
    heroes_completed = parse_int(hero_match.group(1)) if hero_match else None
    heroes_total = parse_int(hero_match.group(2)) if hero_match else None

    deaths, skips = _parse_total_deaths_skips(compact)

    time_match = re.search(
        r"(?:通关)?总(?:计)?(?:时|耗时|扬时|训耗时)\s*"
        r"((?:[0-9OoIlSB]+\s*小时)?\s*"
        r"(?:[0-9OoIlSB]+\s*分)?\s*"
        r"(?:[0-9OoIlSB]+(?:\.[0-9]+)?\s*秒?)?)",
        compact,
    )
    clear_time = time_match.group(1).strip() if time_match else None
    if not clear_time or parse_time_to_seconds(clear_time) is None:
        # The congratulation banner can cover the label and an icon can split it from the value, yet the value is
        # still read. It is the last complete time in the stats block.
        values = list(_CLEAR_TIME_VALUE.finditer(compact))
        clear_time = values[-1].group(0).strip() if values else clear_time
    return LeftPanel(
        heroes_completed=heroes_completed,
        heroes_total=heroes_total,
        challenge_completed=True if "挑战完成" in compact else None,
        achievement_challenge_disabled=True if ACHIEVEMENT_DISABLED.search(compact) else None,
        total_deaths=deaths,
        total_skips=skips,
        clear_time=clear_time,
        clear_time_seconds=parse_time_to_seconds(clear_time or "") if clear_time else None,
    )


# The game prints this notice when a setting makes achievements unobtainable; "已" and "用" are the glyphs OCR drops.
ACHIEVEMENT_DISABLED = re.compile(r"成就挑战\s*已?\s*禁")

_CLEAR_TIME_VALUE = re.compile(
    r"(?:[0-9OoIlSB]+\s*小时\s*)?[0-9OoIlSB]+\s*分\s*[0-9OoIlSB]+(?:\.[0-9]+)?\s*秒"
)

_ACHIEVEMENT_CHECK = re.compile(
    r"(?P<title>[^✓✔√☑☒\n]+?)\s*"
    r"(?:[✓✔√☑☒]|[VvLl](?=\s|$))(?=\s|$)"
)


def parse_achievement_titles(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            match.group("title").strip()
            for match in _ACHIEVEMENT_CHECK.finditer(text)
            if match.group("title").strip()
        )
    )


def _parse_total_deaths_skips(compact: str) -> tuple[int | None, int | None]:
    normalized = compact.translate(str.maketrans({"／": "/", "｜": "/", "|": "/", "%": "/", " ": ""}))

    anchor = re.search(r"总计(?:死亡|阵亡|车二)(?:/跳过|跳过|过)?", normalized)
    if not anchor:
        unlabeled = re.search(r"(?<![0-9])([0-9]+)\s*[/\.]\s*([0-9]+)\s*(?=增益|减益)", normalized)
        if not unlabeled:
            return None, None
        return parse_int(unlabeled.group(1)), parse_int(unlabeled.group(2))

    nearby = normalized[anchor.end() : anchor.end() + 28].split("增益", 1)[0]
    nearby = nearby.replace("次", "")
    nearby = nearby.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")

    m = re.search(r"([0-9]+)\s*[/\.]\s*([0-9]+)", nearby)
    if not m:
        return None, None

    return parse_int(m.group(1)), parse_int(m.group(2))
