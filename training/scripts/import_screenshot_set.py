from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.importer.client import HttpScreenshotSetClient
from training.importer.importer import import_screenshot_set
from training.studio.core import DEFAULT_WORK_ROOT, create_batch, roi_preview_paths
from training.studio.r2 import StudioR2Store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import one finalized platform screenshot set as a reviewable Studio batch."
    )
    parser.add_argument("--version", required=True, type=int, help="Finalized screenshot set version to import")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="Studio work root for the new batch")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Resumable download/verification cache (default: <work-root>/set-workspace/set-<version>)",
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true", help="Re-download and re-verify all set members")
    parser.add_argument("--code-revision", default=None, help="Override the recorded OCRKit code revision")
    args = parser.parse_args()

    base_url = os.environ.get("OCRKIT_SCREENSHOT_SET_BASE_URL", "").strip()
    token = os.environ.get("OCRKIT_SCREENSHOT_SET_TOKEN", "").strip()
    if not base_url:
        raise SystemExit("OCRKIT_SCREENSHOT_SET_BASE_URL is required")
    if not token:
        raise SystemExit("OCRKIT_SCREENSHOT_SET_TOKEN is required")
    store = StudioR2Store.from_settings()
    if store is None:
        raise SystemExit("Studio R2 is not configured (OCRKIT_R2_* / OCRKIT_STUDIO_R2_*)")

    work_root = args.work_root.resolve()
    workspace = args.workspace or (work_root / "set-workspace" / f"set-{args.version}")
    client = HttpScreenshotSetClient(base_url, token, timeout_seconds=args.timeout)
    metadata = client.fetch_set(args.version)

    report = import_screenshot_set(
        metadata=metadata,
        download_object=lambda key: store.get_image(key)[0],
        workspace=workspace,
        expected_version=args.version,
        resume=not args.no_resume,
        code_revision=args.code_revision,
    )
    batch_dir, summary = create_batch(
        report.member_files,
        work_root=work_root,
        holdout_ratio=args.holdout_ratio,
        provenance_by_digest=report.provenance_by_digest,
        screenshot_set=report.screenshot_set,
    )
    roi_preview_paths(batch_dir)
    print(
        json.dumps(
            {
                "set_id": report.set_id,
                "version": report.version,
                "member_count": report.member_count,
                "accuracy_counts": report.accuracy_counts,
                "batch": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"screenshot set v{report.version} imported as Studio batch {summary['batch_id']}")


if __name__ == "__main__":
    main()
