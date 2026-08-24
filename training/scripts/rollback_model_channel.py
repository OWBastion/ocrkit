from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.release import STABLE_CHANNEL_KEY, parse_channel, parse_manifest, validate_channel_key
from training.scripts.promote_model_channel import verify_manifest


def _client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )


def rollback(client: Any, bucket: str, stable_channel_key: str, manifest_key: str) -> dict[str, Any]:
    validate_channel_key(stable_channel_key, allow_stable=True)
    stable_payload = client.get_object(Bucket=bucket, Key=stable_channel_key)["Body"].read()
    stable = parse_channel(stable_payload, stable_channel_key)
    current_key = str(stable["manifest_key"])
    history = stable.get("history") if isinstance(stable.get("history"), list) else []
    allowed = {entry.get("manifest_key") for entry in history if isinstance(entry, dict)}
    if manifest_key == current_key or manifest_key not in allowed:
        raise ValueError("rollback target is not a previously verified stable manifest")
    manifest_payload = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    parse_manifest(manifest_payload, manifest_key)
    verify_manifest(bucket, manifest_key)
    payload = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "manifest_key": manifest_key,
        "previous_manifest_key": current_key,
        "history": [*history, {"manifest_key": current_key, "verified_at": stable.get("updated_at"), "action": "rollback-source"}],
        "action": "rollback",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    client.put_object(
        Bucket=bucket,
        Key=stable_channel_key,
        Body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        ContentType="application/json",
    )
    return {"rolled_back": True, "manifest_key": manifest_key, "previous_manifest_key": current_key}


def main() -> None:
    parser = argparse.ArgumentParser(description="Repoint stable to a previously verified manifest.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--candidate-channel")
    parser.add_argument("--stable-channel", default=STABLE_CHANNEL_KEY)
    args = parser.parse_args()
    print(json.dumps(rollback(_client(), args.bucket, args.stable_channel, args.manifest_key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
