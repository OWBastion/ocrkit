from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parser.terminology import (
    TerminologyCatalog,
    default_terminology_catalog,
    normalize_roi_text,
)


def canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def _load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        crop, text = line.split("\t", 1)
        labels[crop] = canonicalize(text)
    return labels


def _load_review(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _terminology_metrics(
    rows: list[dict[str, object]],
    truth: dict[str, str],
    catalog: TerminologyCatalog,
) -> dict[str, object]:
    engines = ("rapidocr", "vision", "teacher")
    raw_scores = {name: {"covered": 0, "correct": 0} for name in engines}
    normalized_scores = {name: {"covered": 0, "correct": 0} for name in engines}
    adopted = 0
    corrected = 0
    false_corrections = 0
    decisions = {"normalized": 0, "unchanged": 0, "ambiguous": 0, "unresolved": 0}

    for row in rows:
        crop = row.get("crop")
        if not isinstance(crop, str) or crop not in truth:
            continue
        expected = truth[crop]
        layout_version = str(row.get("layout_version") or "1280x720-v6")
        roi = str(row.get("roi") or "")
        decision_counted = False
        for engine in engines:
            text = row.get(f"{engine}_text")
            if not isinstance(text, str):
                continue
            result = normalize_roi_text(text, layout_version, roi, catalog)
            raw_scores[engine]["covered"] += 1
            raw_scores[engine]["correct"] += canonicalize(text) == expected
            normalized_scores[engine]["covered"] += 1
            normalized_scores[engine]["correct"] += canonicalize(result.normalized_text) == expected
            if not decision_counted:
                decisions[result.decision] += 1
                decision_counted = True
            if any(token.status == "adopted" for token in result.tokens):
                adopted += 1
                if canonicalize(text) != expected and canonicalize(result.normalized_text) == expected:
                    corrected += 1
                elif canonicalize(result.normalized_text) != expected:
                    false_corrections += 1

    return {
        "rules_version": catalog.rules_version,
        "raw": raw_scores,
        "normalized": normalized_scores,
        "adopted": adopted,
        "corrected": corrected,
        "false_corrections": false_corrections,
        "hit_rate": corrected / adopted if adopted else 0.0,
        "false_correction_rate": false_corrections / adopted if adopted else 0.0,
        "decisions": decisions,
    }


def evaluate(
    output_dir: Path,
    terminology_catalog: TerminologyCatalog | None = None,
) -> dict[str, dict[str, int] | dict[str, object]]:
    truth = _load_labels(output_dir / "labels/holdout.txt")
    rows = _load_review(output_dir / "review/holdout.jsonl")
    scores = {name: {"correct": 0, "covered": 0} for name in ("rapidocr", "vision", "teacher", "agreement")}

    for row in rows:
        crop = row.get("crop")
        if not isinstance(crop, str) or crop not in truth:
            continue
        expected = truth[crop]
        rapid = row.get("rapidocr_text")
        vision = row.get("vision_text")
        if isinstance(rapid, str):
            scores["rapidocr"]["covered"] += 1
            scores["rapidocr"]["correct"] += canonicalize(rapid) == expected
        if isinstance(vision, str):
            scores["vision"]["covered"] += 1
            scores["vision"]["correct"] += canonicalize(vision) == expected
        teacher = row.get("teacher_text")
        if isinstance(teacher, str):
            scores["teacher"]["covered"] += 1
            scores["teacher"]["correct"] += canonicalize(teacher) == expected
        results = [text for text in (rapid, vision, teacher) if isinstance(text, str) and text.strip()]
        if len(results) >= 2 and len({canonicalize(text) for text in results}) == 1:
            scores["agreement"]["covered"] += 1
            scores["agreement"]["correct"] += canonicalize(results[0]) == expected

    result: dict[str, dict[str, int] | dict[str, object]] = {"labels": {"total": len(truth)}, **scores}
    if terminology_catalog is not None:
        result["terminology"] = _terminology_metrics(rows, truth, terminology_catalog)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RapidOCR, Apple Vision, and the previous-model candidates against reviewed holdout labels.")
    parser.add_argument("--output", type=Path, default=Path("datasets/labeled/rec"))
    parser.add_argument("--without-terminology", action="store_true", help="Skip terminology-normalized accuracy metrics")
    args = parser.parse_args()
    catalog = None if args.without_terminology else default_terminology_catalog()
    print(json.dumps(evaluate(args.output, catalog), ensure_ascii=False))


if __name__ == "__main__":
    main()
