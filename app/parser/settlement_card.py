from __future__ import annotations

from dataclasses import dataclass, field
import re

from .left_panel import ACHIEVEMENT_DISABLED
from .right_panel import resolve_map

_SEP = r"\s*[·.•]?\s*"
_TITLE = re.compile(rf"挑战完成{_SEP}(?P<h>\d{{1,3}}):(?P<m>\d{{2}}):(?P<s>\d{{2}})")
_PLAYER = re.compile(r"玩家\s*(?P<player>.+?)\s*[·•]\s*通关\s*代码")
_STATS = re.compile(rf"阵亡\s*(?P<deaths>\d+){_SEP}跳过\s*(?P<skips>\d+)")
_DIFFICULTY = re.compile(r"地狱|传奇|困难|普通|简单|无法跳过英雄")
_VERSION = re.compile(r"[vV]\s*(?P<version>\d{2}\.\d{4}\.\d+)")
_RUN_CODE = re.compile(r"通关\s*代码\s*[:：]?\s*[1-9]\d{3}\s*[-‐‑‒–—−－]\s*\d{4}\s*[-‐‑‒–—−－]\s*\d{4}")
_TITLES = re.compile(r"称号\s*(?P<titles>.*?)\s*(?:剩余.*)?$")
_TITLE_SEPARATOR = re.compile(r"\s*[·•]\s*")
# An empty title line prints an em dash, which OCR often reads as a hyphen or the character "一".
_NO_TITLES = re.compile(r"^[—–\-一]*$")


@dataclass
class SettlementCard:
    completed: bool = False
    duration_text: str | None = None
    duration_seconds: float | None = None
    player: str | None = None
    deaths: int | None = None
    skips: int | None = None
    map_name: str | None = None
    map_variant: str | None = None
    difficulty: str | None = None
    version: str | None = None
    titles: list[str] = field(default_factory=list)
    achievement_challenge_disabled: bool | None = None


def parse_settlement_card(text: str, map_names: list[str], map_aliases: dict[str, str] | None = None) -> SettlementCard:
    """Read the completion card Bastion shows for 20 seconds at the top centre after a run."""
    cleaned = text.replace("|", " ")
    card = SettlementCard()

    title = _TITLE.search(cleaned)
    if title is None:
        return card
    card.completed = True
    hours, minutes, seconds = int(title["h"]), int(title["m"]), int(title["s"])
    card.duration_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    card.duration_seconds = float(hours * 3600 + minutes * 60 + seconds)

    if player := _PLAYER.search(cleaned):
        card.player = player["player"]
    if stats := _STATS.search(cleaned):
        card.deaths, card.skips = int(stats["deaths"]), int(stats["skips"])

    code = _RUN_CODE.search(cleaned)
    difficulty = _DIFFICULTY.search(cleaned, code.end() if code else title.end())
    if difficulty is not None:
        card.difficulty = difficulty.group()
        map_text = cleaned[code.end() if code else title.end() : difficulty.start()].strip(" ·.•")
        if map_text:
            card.map_name, card.map_variant = resolve_map(map_text, map_names, map_aliases, map_text)
    if version := _VERSION.search(cleaned, difficulty.end() if difficulty else title.end()):
        card.version = version["version"]

    if titles := _TITLES.search(cleaned):
        value = titles["titles"].strip()
        if ACHIEVEMENT_DISABLED.search(value):
            card.achievement_challenge_disabled = True
        elif not _NO_TITLES.fullmatch(value):
            card.titles = [item for item in _TITLE_SEPARATOR.split(value) if item]
    return card
