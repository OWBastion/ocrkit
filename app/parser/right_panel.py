from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

_DIFFICULTY = re.compile(r"(地狱|传奇|困难|普通|简单|无法跳过英雄)")
_VERSION = re.compile(r"(?:版本|股本)\s*([0-9\.OoIlSB]+)")
_MAP_SPLIT = re.compile(r"[：:]")
_MAP_TEXT_CLEAN = str.maketrans(
    {
        " ": "",
        "\t": "",
        "\n": "",
        "\r": "",
        "，": "",
        ",": "",
        "。": "",
        ".": "",
        "·": "",
        "-": "",
        "_": "",
    }
)
_MAP_COMPAT_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})


@dataclass
class RightPanel:
    map_name: str | None
    difficulty: str | None
    version: str | None


def _normalize_map_text(text: str) -> str:
    return text.translate(_MAP_TEXT_CLEAN).strip()


def _compat_map_text(text: str) -> str:
    return _normalize_map_text(text).translate(_MAP_COMPAT_FIX)


def _map_candidate_before_difficulty(raw_text: str, difficulty_match: re.Match[str] | None) -> str | None:
    if difficulty_match is None:
        return None
    left_context = raw_text[: difficulty_match.start()]
    if not left_context:
        return None
    last_colon = max(left_context.rfind("："), left_context.rfind(":"))
    if last_colon < 0:
        return None
    head = left_context[:last_colon]
    split_pos = max(
        head.rfind(" "),
        head.rfind("\t"),
        head.rfind("\n"),
        head.rfind("["),
        head.rfind("]"),
        head.rfind("【"),
        head.rfind("】"),
    )
    candidate = head[split_pos + 1 :] if split_pos >= 0 else head[max(0, len(head) - 24) :]
    candidate = _normalize_map_text(candidate)
    return candidate or None


def _best_map(raw_text: str, map_names: list[str]) -> str | None:
    if not map_names:
        return None

    map_candidates = [raw_text]
    split = _MAP_SPLIT.split(raw_text, maxsplit=1)
    if split:
        map_candidates.insert(0, split[0])

    normalized_to_raw: dict[str, str] = {}
    compat_to_raw: dict[str, str] = {}
    for item in map_names:
        normalized = _normalize_map_text(item)
        compat = _compat_map_text(item)
        if normalized and normalized not in normalized_to_raw:
            normalized_to_raw[normalized] = item
        if compat and compat not in compat_to_raw:
            compat_to_raw[compat] = item

    for candidate in map_candidates:
        normalized_candidate = _normalize_map_text(candidate)
        if normalized_candidate in normalized_to_raw:
            return normalized_to_raw[normalized_candidate]

        compat_candidate = _compat_map_text(candidate)
        if compat_candidate in compat_to_raw:
            return compat_to_raw[compat_candidate]

    best_name = None
    best_score = 0.0
    for candidate in map_candidates:
        compat_candidate = _compat_map_text(candidate)
        if not compat_candidate:
            continue
        for compat_name, raw_name in compat_to_raw.items():
            score = SequenceMatcher(None, compat_candidate, compat_name).ratio()
            if score > best_score:
                best_name = raw_name
                best_score = score

    return best_name if best_score >= 0.6 else None


def parse_right_panel(text: str, map_names: list[str]) -> RightPanel:
    difficulty = None
    m = _DIFFICULTY.search(text)
    if m:
        difficulty = m.group(1)
    anchored_map = _map_candidate_before_difficulty(text, m)
    map_name = _best_map(anchored_map, map_names) if anchored_map else None
    if map_name is None:
        map_name = _best_map(text, map_names)

    version = None
    vm = _VERSION.search(text)
    if vm:
        version = vm.group(1).translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}))

    return RightPanel(
        map_name=map_name,
        difficulty=difficulty,
        version=version,
    )
