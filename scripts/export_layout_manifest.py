from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_manifest(source: Path) -> dict[str, object]:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    standard_size = data["standard_size"]
    return {
        "schema_version": "1",
        "layout_version": str(data["layout_version"]),
        "standard_size": {
            "width": int(standard_size["width"]),
            "height": int(standard_size["height"]),
        },
        "rois": {
            name: {key: int(item[key]) for key in ("x1", "y1", "x2", "y2")}
            for name, item in data["rois"].items()
        },
    }


def render(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the authoritative OCRKit ROI YAML as JSON.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true", help="fail when output is not up to date")
    args = parser.parse_args()

    expected = render(build_manifest(args.source))
    if args.check:
        actual = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if actual != expected:
            raise SystemExit(f"layout manifest is out of date: {args.output}")
        return

    args.output.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
