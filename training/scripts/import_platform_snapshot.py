from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.importer.client import HttpSnapshotClient
from training.importer.importer import import_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import one finalized platform-reviewed dataset snapshot for OCRKit preparation."
    )
    parser.add_argument("--snapshot-id", required=True, help="Platform snapshot identity to import")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Resumable download/verification cache (default: training/.work/imports/<snapshot-id>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Materialized import directory (default: datasets/labeled/rec/platform/<snapshot-id>@<version>)",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="ocrkit-v1")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true", help="Re-download and re-verify all snapshot objects")
    parser.add_argument("--code-revision", default=None, help="Override the recorded OCRKit code revision")
    args = parser.parse_args()

    base_url = os.environ.get("OCRKIT_PLATFORM_SNAPSHOT_BASE_URL", "").strip()
    token = os.environ.get("OCRKIT_PLATFORM_SNAPSHOT_TOKEN", "").strip()
    if not base_url:
        raise SystemExit("OCRKIT_PLATFORM_SNAPSHOT_BASE_URL is required")
    if not token:
        raise SystemExit("OCRKIT_PLATFORM_SNAPSHOT_TOKEN is required")

    workspace = args.workspace or (ROOT / f"training/.work/imports/{args.snapshot_id}")
    # The output defaults to the finalized private datasets submodule location;
    # the version suffix is only known after the snapshot metadata is fetched.
    client = HttpSnapshotClient(base_url, token, timeout_seconds=args.timeout)
    metadata = client.fetch_snapshot(args.snapshot_id)
    output = args.output or (ROOT / f"datasets/labeled/rec/platform/{args.snapshot_id}@{metadata.version}")

    report = import_snapshot(
        client=client,
        snapshot_id=args.snapshot_id,
        workspace=workspace,
        output=output,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
        resume=not args.no_resume,
        code_revision=args.code_revision,
    )
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    print(f"materialized import written to {output}")


if __name__ == "__main__":
    main()
