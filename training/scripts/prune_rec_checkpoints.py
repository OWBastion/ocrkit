from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prune_checkpoint_dir(directory: Path) -> list[Path]:
    """Keep latest and best_accuracy; drop per-epoch dumps and Paddle's duplicate best_model."""
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
    removed: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name.startswith("iter_epoch_") or path.name == "best_model":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
    return removed


def discover_checkpoint_dirs(root: Path) -> list[Path]:
    found: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.name.startswith("iter_epoch_"):
            found.add(path.parent)
        elif path.is_dir() and path.name == "best_model":
            found.add(path.parent)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove PaddleOCR per-epoch dumps and duplicate best_model directories."
    )
    parser.add_argument("directories", nargs="*", type=Path, help="checkpoint directories to prune")
    parser.add_argument("--root", type=Path, help="discover and prune checkpoint directories under this root")
    args = parser.parse_args()
    directories = list(args.directories)
    if args.root is not None:
        directories.extend(discover_checkpoint_dirs(args.root))
    if not directories:
        raise SystemExit("pass checkpoint directories or --root")
    unique = sorted({path.resolve() for path in directories})
    removed: list[Path] = []
    for directory in unique:
        removed.extend(prune_checkpoint_dir(directory))
    for path in removed:
        print(path)


if __name__ == "__main__":
    main()
