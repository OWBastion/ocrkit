from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


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


def evaluate(output_dir: Path) -> dict[str, dict[str, int]]:
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

    return {"labels": {"total": len(truth)}, **scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RapidOCR, Apple Vision, and the previous-model candidates against reviewed holdout labels.")
    parser.add_argument("--output", type=Path, default=Path("datasets/labeled/rec"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
