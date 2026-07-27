from __future__ import annotations

from dataclasses import dataclass

from .normalize import normalize_player_name


@dataclass
class BottomLeftHero:
    player: str | None
    achievement_title: str | None = None


def parse_bottom_left_hero(text: str) -> BottomLeftHero:
    parts = text.split()
    title = normalize_player_name(parts[-1]) if parts else None
    return BottomLeftHero(player=title, achievement_title=title)
