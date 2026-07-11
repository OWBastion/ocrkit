from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_review(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _finalize_split(rows: list[dict[str, object]], destination: Path) -> int:
    accepted: list[str] = []
    for row in rows:
        status = row.get("review_status")
        if status == "rejected":
            continue
        if status != "accepted":
            raise ValueError("all review rows must be accepted or rejected before training")
        crop = row.get("crop")
        transcription = row.get("transcription")
        if not isinstance(crop, str) or not isinstance(transcription, str) or not transcription.strip():
            raise ValueError("accepted review rows require crop and transcription")
        accepted.append(f"{crop}\t{transcription.strip()}")
    if not accepted:
        raise ValueError(f"no accepted labels for {destination.stem}")
    destination.write_text("\n".join(accepted) + "\n", encoding="utf-8")
    return len(accepted)


def finalize(output_dir: Path) -> dict[str, int]:
    review_dir = output_dir / "review"
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    train = _finalize_split(_load_review(review_dir / "train.jsonl"), labels_dir / "train.txt")
    holdout = _finalize_split(_load_review(review_dir / "holdout.jsonl"), labels_dir / "holdout.txt")
    return {"train_labels": train, "holdout_labels": holdout}


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn reviewed rec candidates into PaddleOCR label files.")
    parser.add_argument("--output", type=Path, default=Path("datasets/labeled/rec"))
    args = parser.parse_args()
    print(json.dumps(finalize(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
