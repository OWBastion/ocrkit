from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts import ModelArtifactStore
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.storage.r2_client import R2ObjectStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a published OCRKit model artifact from R2.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--manifest-key", required=True)
    args = parser.parse_args()

    store = R2ObjectStore.from_settings(
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
        default_bucket=args.bucket,
        allowed_buckets_raw=args.bucket,
        read_timeout_seconds=30,
    )
    with tempfile.TemporaryDirectory(prefix="ocrkit-model-verify-") as cache_dir:
        artifacts = ModelArtifactStore(store, args.bucket, Path(cache_dir)).prepare(args.manifest_key)
        RapidOcrEngine(artifacts.rapidocr_config_path)
    print(artifacts.version)


if __name__ == "__main__":
    main()
