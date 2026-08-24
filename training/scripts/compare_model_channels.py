from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import boto3

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.model_artifacts.release import compare_manifests, parse_channel, parse_manifest, validate_channel_key


def _client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
    )


def _get_json(client: Any, bucket: str, key: str) -> tuple[bytes, dict[str, Any]]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body, json.loads(body)


def compare_channels(client: Any, bucket: str, candidate_channel_key: str, stable_channel_key: str) -> dict[str, Any]:
    validate_channel_key(candidate_channel_key)
    validate_channel_key(stable_channel_key, allow_stable=True)
    candidate_channel_bytes, _ = _get_json(client, bucket, candidate_channel_key)
    stable_channel_bytes, _ = _get_json(client, bucket, stable_channel_key)
    candidate_channel = parse_channel(candidate_channel_bytes, candidate_channel_key)
    stable_channel = parse_channel(stable_channel_bytes, stable_channel_key)
    candidate_key = str(candidate_channel["manifest_key"])
    stable_key = str(stable_channel["manifest_key"])
    _, candidate_manifest = _get_json(client, bucket, candidate_key)
    _, stable_manifest = _get_json(client, bucket, stable_key)
    candidate_manifest = parse_manifest(json.dumps(candidate_manifest).encode(), candidate_key)
    stable_manifest = parse_manifest(json.dumps(stable_manifest).encode(), stable_key)
    report = compare_manifests(candidate_key, candidate_manifest, stable_key, stable_manifest)
    report["candidate"]["channel_key"] = candidate_channel_key
    report["stable"]["channel_key"] = stable_channel_key
    report["stable"]["history"] = stable_channel.get("history", [])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the verified candidate channel with the current stable manifest.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--candidate-channel", default="models/pp-ocrv6-small/channels/candidate.json")
    parser.add_argument("--stable-channel", default="models/pp-ocrv6-small/channels/stable.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fail-on-ineligible", action="store_true")
    args = parser.parse_args()
    report = compare_channels(_client(), args.bucket, args.candidate_channel, args.stable_channel)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_ineligible and not report["eligible"]:
        raise SystemExit("candidate is not eligible for promotion: " + "; ".join(report["reasons"]))


if __name__ == "__main__":
    main()
