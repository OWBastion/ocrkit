from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


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
    parser.add_argument("--prefix", default="models/pp-ocrv6-small")
    args = parser.parse_args()

    if not VERSION_RE.fullmatch(args.version):
        raise SystemExit("version may contain only letters, digits, dots, underscores, and hyphens")
    if not args.artifact_dir.is_dir():
        raise SystemExit(f"artifact directory does not exist: {args.artifact_dir}")

    object_prefix = f"{args.prefix.strip('/')}/{args.version}"
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
