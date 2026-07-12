from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.constants import MODEL_OBJECT_PREFIX


def _occupied(client: object, bucket: str, version: str) -> bool:
    key = f"{MODEL_OBJECT_PREFIX}/{version}/manifest.json"
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an unused UTC OCR model version.")
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )
    base = dt.datetime.now(dt.timezone.utc).strftime("%Y.%m.%d-%H%M%S")
    version = base
    suffix = 0
    while _occupied(client, args.bucket, version):
        suffix += 1
        version = f"{base}-{suffix:02d}"
    print(version)


if __name__ == "__main__":
    main()
