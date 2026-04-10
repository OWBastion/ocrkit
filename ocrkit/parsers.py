from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


PASS_PATTERN = re.compile(
    r"祝贺\s*(?P<name>.+?)\s*以\s*(?P<deaths>\d+)\s*次阵亡\s*[&＆]\s*(?P<skips>\d+)\s*次跳过\s*耗时\s*(?P<time>.+?)\s*通关"
)
TIME_HMS_PATTERN = re.compile(
    r"(?:(?P<h>\d+)\s*小时)?\s*(?:(?P<m>\d+)\s*分)?\s*(?:(?P<s>\d+(?:\.\d+)?)\s*秒)?"
)
TIME_MMSS_PATTERN = re.compile(r"(?P<m>\d{1,3}):(?P<s>\d{1,2}(?:\.\d+)?)")
TIME_SEC_PATTERN = re.compile(r"(?P<s>\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
DIFFICULTY_PATTERN = re.compile(r"(地狱|传奇|困难|普通|简单)")


@dataclass
class ParsedRun:
    passed: bool
    player_name: str | None
    time_sec: float | None
    deaths: int | None
    skips: int | None
    difficulty: str | None
    map_label: str | None


def parse_time_to_seconds(value: str) -> float | None:
    text = value.replace("：", ":").strip()

    m = TIME_MMSS_PATTERN.search(text)
    if m:
        minutes = int(m.group("m"))
        seconds = float(m.group("s"))
        return round(minutes * 60 + seconds, 2)

    m = TIME_SEC_PATTERN.search(text)
    if m:
        return round(float(m.group("s")), 2)

    m = TIME_HMS_PATTERN.search(text)
    if m and any(m.group(k) for k in ("h", "m", "s")):
        hours = int(m.group("h") or 0)
        minutes = int(m.group("m") or 0)
        seconds = float(m.group("s") or 0)
        return round(hours * 3600 + minutes * 60 + seconds, 2)

    return None


def parse_difficulty(text: str) -> str | None:
    m = DIFFICULTY_PATTERN.search(text)
    return m.group(1) if m else None


def match_map_label(texts: list[str], labels: list[str]) -> tuple[str | None, float]:
    joined = " ".join(texts)
    best_label: str | None = None
    best_score = 0.0

    for label in labels:
        if label in joined:
            return label, 1.0
        score = SequenceMatcher(None, joined, label).ratio()
        if score > best_score:
            best_label = label
            best_score = score

    if best_score >= 0.45:
        return best_label, best_score
    return None, best_score


def parse_run_from_texts(center_banner: str, top_bar: str, left_panel: str, map_labels: list[str]) -> ParsedRun:
    merged_for_meta = f"{top_bar}\n{left_panel}"
    difficulty = parse_difficulty(merged_for_meta)
    map_label, _ = match_map_label([top_bar, left_panel], map_labels)

    m = PASS_PATTERN.search(center_banner.replace("|", ""))
    if not m:
        passed = "通关" in center_banner
        return ParsedRun(
            passed=passed,
            player_name=None,
            time_sec=None,
            deaths=None,
            skips=None,
            difficulty=difficulty,
            map_label=map_label,
        )

    player_name = m.group("name").strip()
    deaths = int(m.group("deaths"))
    skips = int(m.group("skips"))
    time_sec = parse_time_to_seconds(m.group("time"))

    return ParsedRun(
        passed=True,
        player_name=player_name,
        time_sec=time_sec,
        deaths=deaths,
        skips=skips,
        difficulty=difficulty,
        map_label=map_label,
    )
