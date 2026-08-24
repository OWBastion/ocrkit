from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.constants import MODEL_OBJECT_PREFIX
from app.model_artifacts.release import validate_channel_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically update an OCRKit model release channel after verification.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--channel-key", required=True)
    parser.add_argument("--manifest-key", required=True)
    args = parser.parse_args()
    try:
        validate_channel_key(args.channel_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.manifest_key.startswith(f"{MODEL_OBJECT_PREFIX}/") or not args.manifest_key.endswith("/manifest.json"):
        raise SystemExit("model manifest key is invalid")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )
    payload = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "manifest_key": args.manifest_key,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    client.put_object(
        Bucket=args.bucket,
        Key=args.channel_key,
        Body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        ContentType="application/json",
    )
    print(json.dumps({"bucket": args.bucket, "channel_key": args.channel_key, "manifest_key": args.manifest_key}, ensure_ascii=False))


if __name__ == "__main__":
    main()
