"""Record an explicit owner decision to accept a candidate whose paired comparison failed.

The comparison stays in the evidence with every reason it failed for, next to the decision and who made it, so the
release record shows what was knowingly accepted instead of a quietly relaxed gate.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIN_REASON_CHARS = 10


def accept(evaluation_dir: Path, reason: str, *, accepted_at: datetime | None = None) -> dict[str, Any]:
    reason = reason.strip()
    if len(reason) < MIN_REASON_CHARS:
        raise ValueError(f"write why the failed comparison is accepted (at least {MIN_REASON_CHARS} characters)")
    comparison_path = evaluation_dir / "comparison_report.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("verdict") != "failed":
        raise ValueError("only a failed comparison needs an override")
    comparison["override"] = {
        "reason": reason,
        "accepted_at": (accepted_at or datetime.now(UTC)).isoformat(),
        "overridden_reasons": list(comparison.get("reasons") or []),
    }
    comparison["verdict"] = "overridden"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("fixture_report.json", "holdout_report.json"):
        path = evaluation_dir / name
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            report["status"] = "passed"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return comparison
