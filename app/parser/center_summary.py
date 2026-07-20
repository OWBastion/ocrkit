from __future__ import annotations

from dataclasses import dataclass
import re

from .normalize import parse_time_to_seconds


_CENTER_PATTERNS = [
    re.compile(
        r"(?:祝)?贺\s*.+?\s*(?:以\s*)?(?P<deaths>\d+)\s*次阵亡\s*[&＆]\s*(?P<skips>\d+)\s*次跳过\s*[·.]?\s*耗时\s*(?P<time>.+?)\s*通关"
    )
]


@dataclass
class CenterSummary:
    completed: bool
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
            deaths=int(m.group("deaths")),
            skips=int(m.group("skips")),
            duration_text=duration_text,
            duration_seconds=parse_time_to_seconds(duration_text),
        )

    return CenterSummary(
        completed=("通关" in cleaned),
        deaths=None,
        skips=None,
        duration_text=None,
        duration_seconds=None,
    )
