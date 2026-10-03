from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.ocr.engine import OcrChunk

from .normalize import normalize_player_name

# Health and ammo counters share the hero card with the name plate.
_COUNTER = re.compile(r"[\d\s/|.,:+\-%]+")
# The player-level badge is read as a digit in front of the name ("3初一", "3 训犬大师").
_LEVEL_BADGE_PREFIX = re.compile(r"\d\s*(?=[^\d\s])")


@dataclass
class BottomLeftHero:
    player: str | None


def _name_chunk(chunks: Sequence[OcrChunk]) -> OcrChunk | None:
    boxed = [chunk for chunk in chunks if chunk.box is not None and chunk.text.strip()]
    if not boxed:
        return None
    # The level badge and rank icon are read as short fragments that may come after the name; the name is the widest
    # fragment that is not a counter (rightmost on a tie).
    names = [chunk for chunk in boxed if not _COUNTER.fullmatch(chunk.text.strip())]
    if not names:
        return max(boxed, key=lambda chunk: chunk.box[3])  # type: ignore[index]
    return max(names, key=lambda chunk: (chunk.box[2] - chunk.box[0], chunk.box[2]))  # type: ignore[index]


def parse_bottom_left_hero(text: str, chunks: Sequence[OcrChunk] = ()) -> BottomLeftHero:
    chunk = _name_chunk(chunks)
    if chunk is not None:
        name = chunk.text.strip()
        stripped = _LEVEL_BADGE_PREFIX.sub("", name, count=1) if _LEVEL_BADGE_PREFIX.match(name) else name
        return BottomLeftHero(player=normalize_player_name(stripped) or None)
    parts = text.split()
    return BottomLeftHero(player=normalize_player_name(parts[-1]) if parts else None)
