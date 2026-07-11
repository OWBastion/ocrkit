from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fail(line_number: int, message: str) -> ValueError:
    return ValueError(f"line {line_number}: {message}")


def _image_path(label_file: Path, image_name: str) -> Path:
    direct = label_file.parent / "images" / image_name
    if direct.is_file():
        return direct
    return label_file.parent.parent / image_name


def validate_rec(label_file: Path) -> int:
    count = 0
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            image_name, transcription = line.split("\t", 1)
        except ValueError as exc:
            raise _fail(line_number, "expected image path and transcription separated by a tab") from exc
        if not image_name or not transcription:
            raise _fail(line_number, "image path and transcription must both be non-empty")
        if not _image_path(label_file, image_name).is_file():
            raise _fail(line_number, f"image does not exist: {image_name}")
        count += 1
    return count


def validate_det(label_file: Path) -> int:
    count = 0
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            image_name, raw_annotations = line.split("\t", 1)
            annotations = json.loads(raw_annotations)
        except (ValueError, json.JSONDecodeError) as exc:
            raise _fail(line_number, "expected image path and JSON annotations separated by a tab") from exc
        if not image_name or not isinstance(annotations, list) or not annotations:
            raise _fail(line_number, "annotations must be a non-empty JSON list")
        if not _image_path(label_file, image_name).is_file():
            raise _fail(line_number, f"image does not exist: {image_name}")
        for annotation in annotations:
            if not isinstance(annotation, dict) or not annotation.get("transcription"):
                raise _fail(line_number, "each annotation needs a non-empty transcription")
            points = annotation.get("points")
            if not isinstance(points, list) or len(points) != 4:
                raise _fail(line_number, "each annotation needs exactly four points")
            if any(not isinstance(point, list) or len(point) != 2 for point in points):
                raise _fail(line_number, "each point must be a two-value list")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PaddleOCR label files.")
    parser.add_argument("task", choices=("det", "rec"))
    parser.add_argument("label_file", type=Path)
    args = parser.parse_args()

    if not args.label_file.is_file():
        raise SystemExit(f"label file does not exist: {args.label_file}")
    count = validate_det(args.label_file) if args.task == "det" else validate_rec(args.label_file)
    print(json.dumps({"task": args.task, "valid_samples": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
