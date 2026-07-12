from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.constants import MODEL_OBJECT_PREFIX, model_version_prefix


REQUIRED_FILES = ("det.onnx", "rec.onnx", "rec_dict.txt", "rapidocr.yaml")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned OCRKit model manifest.")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--prefix", default=MODEL_OBJECT_PREFIX)
    args = parser.parse_args()

    if not VERSION_RE.fullmatch(args.version):
        raise SystemExit("version may contain only letters, digits, dots, underscores, and hyphens")
    if not args.artifact_dir.is_dir():
        raise SystemExit(f"artifact directory does not exist: {args.artifact_dir}")

    if args.prefix.strip("/") != MODEL_OBJECT_PREFIX:
        raise SystemExit(f"model prefix is fixed to {MODEL_OBJECT_PREFIX}")
    object_prefix = model_version_prefix(args.version)
    files: dict[str, dict[str, object]] = {}
    for name in REQUIRED_FILES:
        path = args.artifact_dir / name
        if not path.is_file():
            raise SystemExit(f"required artifact is missing: {path}")
        files[name] = {
            "object_key": f"{object_prefix}/{name}",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "version": args.version,
        "files": files,
    }
    destination = args.artifact_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
