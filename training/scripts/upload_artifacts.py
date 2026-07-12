from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.constants import MODEL_OBJECT_PREFIX


def _object_exists(client: object, bucket: str, object_key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=object_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


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
    version = manifest.get("version")
    if not isinstance(version, str) or not version or "/" in version:
        raise SystemExit("manifest has an invalid version")
    expected_prefix = f"{MODEL_OBJECT_PREFIX}/{version}/"

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )
    sources: list[tuple[Path, str]] = []
    for name, metadata in files.items():
        source = args.artifact_dir / name
        object_key = metadata.get("object_key") if isinstance(metadata, dict) else None
        if not source.is_file() or not isinstance(object_key, str):
            raise SystemExit(f"invalid manifest entry: {name}")
        if not object_key.startswith(expected_prefix):
            raise SystemExit(f"model object key is outside {MODEL_OBJECT_PREFIX}")
        sources.append((source, object_key))

    manifest_key = next(iter(files.values()))["object_key"].rsplit("/", 1)[0] + "/manifest.json"
    for object_key in [*(key for _, key in sources), manifest_key]:
        if _object_exists(client, args.bucket, object_key):
            raise SystemExit(f"model artifact already exists: {object_key}")

    for source, object_key in sources:
        client.upload_file(str(source), args.bucket, object_key)
    client.upload_file(str(manifest_path), args.bucket, manifest_key)
    print(json.dumps({"bucket": args.bucket, "manifest_key": manifest_key}, ensure_ascii=False))


if __name__ == "__main__":
    main()
