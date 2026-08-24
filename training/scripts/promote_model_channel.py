from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.release import STABLE_CHANNEL_KEY, parse_channel, validate_channel_key
from app.model_artifacts.store import ModelArtifactStore
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.storage.r2_client import R2ObjectStore
from training.scripts.compare_model_channels import compare_channels


def _client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )


def verify_manifest(bucket: str, manifest_key: str) -> str:
    store = R2ObjectStore.from_settings(
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
        default_bucket=bucket,
        allowed_buckets_raw=bucket,
        read_timeout_seconds=30,
    )
    with tempfile.TemporaryDirectory(prefix="ocrkit-model-promote-") as cache_dir:
        artifacts = ModelArtifactStore(store, bucket, Path(cache_dir)).prepare(manifest_key)
        RapidOcrEngine(artifacts.rapidocr_config_path)
    return artifacts.version


def _write_stable(client: Any, bucket: str, stable_channel_key: str, manifest_key: str, previous: dict[str, Any]) -> None:
    history = previous.get("history") if isinstance(previous.get("history"), list) else []
    previous_key = previous.get("manifest_key")
    if isinstance(previous_key, str) and previous_key != manifest_key:
        history = [*history, {"manifest_key": previous_key, "verified_at": previous.get("updated_at"), "action": previous.get("action", "release")}]
    payload = {
        "schema_version": 1,
        "model": "pp-ocrv6-small",
        "manifest_key": manifest_key,
        "previous_manifest_key": previous_key,
        "history": history,
        "action": "promote",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    client.put_object(
        Bucket=bucket,
        Key=stable_channel_key,
        Body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        ContentType="application/json",
    )


def promote(client: Any, bucket: str, candidate_channel_key: str, stable_channel_key: str) -> dict[str, Any]:
    validate_channel_key(candidate_channel_key)
    validate_channel_key(stable_channel_key, allow_stable=True)
    report = compare_channels(client, bucket, candidate_channel_key, stable_channel_key)
    if not report["eligible"]:
        raise ValueError("candidate is not eligible for promotion: " + "; ".join(report["reasons"]))
    candidate_channel = parse_channel(
        client.get_object(Bucket=bucket, Key=candidate_channel_key)["Body"].read(), candidate_channel_key
    )
    stable_payload = client.get_object(Bucket=bucket, Key=stable_channel_key)["Body"].read()
    stable_channel = parse_channel(stable_payload, stable_channel_key)
    candidate_key = str(candidate_channel["manifest_key"])
    stable_key = str(stable_channel["manifest_key"])
    verify_manifest(bucket, candidate_key)
    verify_manifest(bucket, stable_key)
    _write_stable(client, bucket, stable_channel_key, candidate_key, stable_channel)
    return {"promoted": True, "candidate_manifest_key": candidate_key, "previous_manifest_key": stable_key, "comparison": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly promote an eligible verified candidate to stable.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--candidate-channel", default="models/pp-ocrv6-small/channels/candidate.json")
    parser.add_argument("--stable-channel", default=STABLE_CHANNEL_KEY)
    args = parser.parse_args()
    result = promote(_client(), args.bucket, args.candidate_channel, args.stable_channel)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
