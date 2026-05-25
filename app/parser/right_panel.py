from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


_DIFFICULTY = re.compile(r"(地狱|传奇|困难|普通|简单|无法跳过英雄)")
_VERSION = re.compile(r"版本\s*([0-9\.OoIlSB]+)")


@dataclass
class RightPanel:
    map_name: str | None
    difficulty: str | None
    version: str | None


def _best_map(raw_text: str, map_names: list[str]) -> str | None:
    if not map_names:
        return None
    for item in map_names:
        if item in raw_text:
            return item

    best_name = None
    best_score = 0.0
    for item in map_names:
        score = SequenceMatcher(None, raw_text, item).ratio()
        if score > best_score:
            best_name = item
            best_score = score
    return best_name if best_score >= 0.45 else None


def parse_right_panel(text: str, map_names: list[str]) -> RightPanel:
    difficulty = None
    m = _DIFFICULTY.search(text)
    if m:
        difficulty = m.group(1)

    version = None
    vm = _VERSION.search(text)
    if vm:
        version = vm.group(1).translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}))

    return RightPanel(
        map_name=_best_map(text, map_names),
        difficulty=difficulty,
        version=version,
    )
