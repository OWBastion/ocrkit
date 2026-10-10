from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.ocr.engine import OcrChunk
from app.schemas.response import EventData, EventNumber

MAX_ACHIEVEMENTS = 5

_EVENT_TITLE = re.compile(r"^(?P<name>.+?)\s*[（(]\s*(?P<seconds>\d+)\s*秒\s*[）)]$")
_NUMBER = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|层|秒|米|点|次|倍)?")
_MARKER = "✓✔√☑☒"

# Pixels brighter than this right of a line, in a region this wide (in line heights), form the check icon.
_CHECK_BRIGHTNESS = 225
_CHECK_REGION_HEIGHTS = 2.2
_CHECK_MIN_PIXELS = 15
_CHECK_MAX_FILL = 0.35

# The stack (event, then achievements) runs down from the top of the region with at most a blank line between its
# parts. Its x position varies: achievements are centred under a wide event but left-aligned without one. Text
# elsewhere in the tall region (signs, name plates, chat) neither starts at the top nor continues the stack.
_STACK_MAX_START_Y = 40.0


def _stack_lines(chunks: list[OcrChunk], scale: float) -> list[OcrChunk]:
    column = sorted((chunk for chunk in chunks if chunk.box is not None), key=lambda chunk: chunk.box[1])  # type: ignore[index]
    kept: list[OcrChunk] = []
    previous_bottom: float | None = None
    for chunk in column:
        top, bottom = chunk.box[1] / scale, chunk.box[3] / scale  # type: ignore[index]
        if previous_bottom is None:
            if top > _STACK_MAX_START_Y:
                break
        elif top - previous_bottom > 2.5 * (bottom - top):
            break
        kept.append(chunk)
        previous_bottom = bottom
    return kept


@dataclass(frozen=True)
class EventStack:
    achievements: tuple[str, ...]
    event: EventData | None


def has_check_mark(gray: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """The completion mark is an icon the OCR does not read, so look for a compact bright blob after the line."""
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    height = max(1, y2 - y1)
    region = gray[max(0, y1):max(0, y2), max(0, x2 + 2):max(0, x2 + 2 + int(height * _CHECK_REGION_HEIGHTS))]
    if region.size == 0:
        return False
    bright = int((region > _CHECK_BRIGHTNESS).sum())
    return bright >= _CHECK_MIN_PIXELS and bright / region.size <= _CHECK_MAX_FILL


def _title_of(text: str) -> str:
    return text.strip().rstrip(_MARKER).strip()


def parse_event_stack(chunks: list[OcrChunk], roi_gray: np.ndarray, scale: float) -> EventStack:
    """Split the stack under the stats block: checked lines are achievements, the rest belong to the event."""
    achievements: list[str] = []
    event_lines: list[OcrChunk] = []
    for chunk in _stack_lines(chunks, scale):
        text_marked = chunk.text.strip()[-1:] in _MARKER
        icon_marked = chunk.box is not None and has_check_mark(roi_gray, tuple(value / scale for value in chunk.box))
        if text_marked or icon_marked:
            if _title_of(chunk.text):
                achievements.append(_title_of(chunk.text))
        else:
            event_lines.append(chunk)

    event = None
    start = next((i for i, chunk in enumerate(event_lines) if _EVENT_TITLE.match(chunk.text.strip())), None)
    if start is not None:
        title = _EVENT_TITLE.match(event_lines[start].text.strip())
        assert title is not None
        description = [chunk.text.strip() for chunk in event_lines[start + 1:]]
        numbers = [
            EventNumber(text=match.group(0).strip(), value=float(match.group("value")), unit=match.group("unit"))
            for line in description
            for match in _NUMBER.finditer(line)
        ]
        event = EventData(
            name=title.group("name").strip(),
            duration_seconds=int(title.group("seconds")),
            description=description,
            numbers=numbers,
            text="\n".join([event_lines[start].text.strip(), *description]),
        )
    return EventStack(tuple(dict.fromkeys(achievements))[:MAX_ACHIEVEMENTS], event)
