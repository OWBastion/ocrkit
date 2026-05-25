from __future__ import annotations

import re


_TIME_HMS = re.compile(r"(?:(?P<h>\d+)\s*小时)?\s*(?:(?P<m>\d+)\s*分)?\s*(?:(?P<s>\d+(?:\.\d+)?)\s*秒)?")
_TIME_MMSS = re.compile(r"(?P<m>\d{1,3})[:：](?P<s>\d{1,2}(?:\.\d+)?)")
_NUM_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


def normalize_numeric_text(text: str) -> str:
    return text.translate(_NUM_FIX)


def parse_int(text: str) -> int | None:
    cleaned = normalize_numeric_text(text)
    m = re.search(r"\d+", cleaned)
    return int(m.group()) if m else None


def parse_time_to_seconds(text: str) -> float | None:
    cleaned = normalize_numeric_text(text).strip()
    mmss = _TIME_MMSS.search(cleaned)
    if mmss:
        return round(int(mmss.group("m")) * 60 + float(mmss.group("s")), 2)

    hms = _TIME_HMS.search(cleaned)
    if hms and any(hms.group(k) for k in ("h", "m", "s")):
        hours = int(hms.group("h") or 0)
        minutes = int(hms.group("m") or 0)
        seconds = float(hms.group("s") or 0)
        return round(hours * 3600 + minutes * 60 + seconds, 2)

    return None


def normalize_player_name(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" !！|_")
