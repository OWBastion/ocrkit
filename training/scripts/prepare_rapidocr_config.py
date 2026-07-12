from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Point RapidOCR config at a release artifact directory.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    config["Det"]["model_path"] = str((args.artifact_dir / "det.onnx").resolve())
    config["Rec"]["model_path"] = str((args.artifact_dir / "rec.onnx").resolve())
    config["Rec"]["rec_keys_path"] = str((args.artifact_dir / "rec_dict.txt").resolve())
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
