from __future__ import annotations

from dataclasses import dataclass, replace
import re


RUN_CODE_MIN_CONFIDENCE = 0.9

_LABEL = r"(?:本局\s*代码|run\s*code)"
_GROUP = r"[1-9]\d{3}"
_SEPARATOR = r"[-‐‑‒–—−－]"
_LABEL_PATTERN = re.compile(_LABEL, re.IGNORECASE)
_CANDIDATE_PATTERN = re.compile(
    rf"{_LABEL}\s*[:：]?\s*"
    rf"(?P<raw>(?P<first>{_GROUP})\s*(?P<separator_one>{_SEPARATOR})\s*"
    rf"(?P<second>{_GROUP})\s*(?P<separator_two>{_SEPARATOR})\s*(?P<third>{_GROUP})(?!\d))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedRunCode:
    value: str | None
    status: str
    normalization: tuple[str, ...] = ()
    warning: str | None = None


def parse_run_code(text: str) -> ParsedRunCode:
    labels = list(_LABEL_PATTERN.finditer(text))
    if not labels:
        return ParsedRunCode(value=None, status="missing")

    candidates = list(_CANDIDATE_PATTERN.finditer(text))
    if len(candidates) != len(labels):
        return ParsedRunCode(value=None, status="invalid", warning="run_code.invalid")

    normalized_codes = {
        "-".join(match.group(name) for name in ("first", "second", "third"))
        for match in candidates
    }
    if len(normalized_codes) != 1:
        return ParsedRunCode(value=None, status="ambiguous", warning="run_code.ambiguous")

    candidate = candidates[0]
    raw = candidate.group("raw")
    normalization: list[str] = []
    if candidate.group("separator_one") != "-" or candidate.group("separator_two") != "-":
        normalization.append("separator:canonical-hyphen")
    if re.search(r"\s", raw):
        normalization.append("whitespace:trimmed")

    return ParsedRunCode(
        value=normalized_codes.pop(),
        status="ok",
        normalization=tuple(normalization),
    )


def enforce_run_code_confidence(parsed: ParsedRunCode, confidence: float) -> ParsedRunCode:
    if parsed.value is not None and confidence < RUN_CODE_MIN_CONFIDENCE:
        return replace(
            parsed,
            value=None,
            status="low_confidence",
            warning="run_code.low_confidence",
        )
    return parsed
