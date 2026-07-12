from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the locked PP-OCRv6 detector.")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    filename = lock["filename"]
    url = lock["url"]
    expected = lock["sha256"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cached = args.cache_dir / filename
    if not cached.is_file() or _sha256(cached) != expected:
        temporary = cached.with_suffix(".download")
        subprocess.run(["curl", "-fsSL", "--retry", "2", url, "-o", str(temporary)], check=True)
        if _sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"detector checksum mismatch: {url}")
        temporary.replace(cached)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
