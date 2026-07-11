from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a manifest and its immutable model artifacts to Cloudflare R2.")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    manifest_path = args.artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("manifest has no files")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )
    for name, metadata in files.items():
        source = args.artifact_dir / name
        object_key = metadata["object_key"]
        if not source.is_file() or not isinstance(object_key, str):
            raise SystemExit(f"invalid manifest entry: {name}")
        client.upload_file(str(source), args.bucket, object_key)

    manifest_key = next(iter(files.values()))["object_key"].rsplit("/", 1)[0] + "/manifest.json"
    client.upload_file(str(manifest_path), args.bucket, manifest_key)
    print(json.dumps({"bucket": args.bucket, "manifest_key": manifest_key}, ensure_ascii=False))


if __name__ == "__main__":
    main()
