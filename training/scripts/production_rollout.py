from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


TARGET_SCRIPT = r'''
import hashlib
import json
import os

import boto3

bucket = os.environ["OCRKIT_R2_DEFAULT_BUCKET"]
channel_key = os.getenv("OCRKIT_MODEL_RELEASE_CHANNEL_KEY") or "models/pp-ocrv6-small/channels/stable.json"
client = boto3.client(
    "s3",
    endpoint_url=os.environ["OCRKIT_R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["OCRKIT_R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["OCRKIT_R2_SECRET_ACCESS_KEY"],
    region_name=os.getenv("OCRKIT_R2_REGION_NAME", "auto"),
)
channel_bytes = client.get_object(Bucket=bucket, Key=channel_key)["Body"].read()
channel = json.loads(channel_bytes)
manifest_key = channel["manifest_key"]
manifest_bytes = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
manifest = json.loads(manifest_bytes)
version = manifest["version"]
expected_key = f"models/pp-ocrv6-small/{version}/manifest.json"
if manifest_key != expected_key:
    raise RuntimeError("stable manifest key does not match its version")
print(json.dumps({
    "channel_key": channel_key,
    "manifest_key": manifest_key,
    "model_version": version,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
}, separators=(",", ":")))
'''

HEALTH_SCRIPT = r'''
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
    print(response.read().decode())
'''

SMOKE_SCRIPT = r'''
import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

image_path = Path(os.environ["OCRKIT_ROLLOUT_SMOKE_IMAGE"])
token = os.environ["OCRKIT_API_TOKEN"]
payload = image_path.read_bytes()
boundary = "----ocrkit-rollout-" + uuid.uuid4().hex
body = (
    f"--{boundary}\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"smoke.png\"\r\n"
    "Content-Type: image/png\r\n\r\n"
).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
request = Request(
    "http://127.0.0.1:8000/api/v1/ocr/challenge",
    data=body,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
    method="POST",
)
with urlopen(request, timeout=30) as response:
    print(response.read().decode())
'''


def compose_command(compose_file: Path, env_file: Path, *args: str) -> list[str]:
    return ["docker", "compose", "--env-file", str(env_file), "-f", str(compose_file), *args]


def json_output(stdout: str, label: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"{label} did not return a JSON object")


def validate_target(target: dict[str, Any]) -> dict[str, str]:
    required = ("channel_key", "manifest_key", "model_version", "manifest_sha256")
    if any(not isinstance(target.get(name), str) or not target[name] for name in required):
        raise RuntimeError("stable channel target is incomplete")
    if not target["manifest_key"].startswith("models/pp-ocrv6-small/"):
        raise RuntimeError("stable manifest is outside the OCRKit model prefix")
    return {name: str(target[name]) for name in required}


def validate_health(health: dict[str, Any], target: dict[str, str]) -> None:
    if health.get("ok") is not True:
        raise RuntimeError("OCRKit health check did not report ok")
    if health.get("model_version") != target["model_version"]:
        raise RuntimeError("OCRKit loaded model_version does not match the stable channel target")


def run_checked(command: list[str], *, label: str, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    result = runner(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result.stdout


def read_target(compose_file: Path, env_file: Path, service: str, runner: Callable[..., subprocess.CompletedProcess[str]]) -> dict[str, str]:
    stdout = run_checked(
        compose_command(
            compose_file,
            env_file,
            "run",
            "--rm",
            "--no-deps",
            "--quiet-pull",
            "--entrypoint",
            "python",
            service,
            "-c",
            TARGET_SCRIPT,
        ),
        label="stable channel read",
        runner=runner,
    )
    return validate_target(json_output(stdout, "stable channel read"))


def read_health(compose_file: Path, env_file: Path, service: str, runner: Callable[..., subprocess.CompletedProcess[str]]) -> dict[str, Any]:
    return json_output(
        run_checked(
            compose_command(compose_file, env_file, "exec", "-T", service, "python", "-c", HEALTH_SCRIPT),
            label="health check",
            runner=runner,
        ),
        "health check",
    )


def wait_for_health(
    compose_file: Path,
    env_file: Path,
    service: str,
    target: dict[str, str],
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "service did not become healthy"
    while time.monotonic() < deadline:
        try:
            health = read_health(compose_file, env_file, service, runner)
            validate_health(health, target)
            return health
        except RuntimeError as exc:
            last_error = str(exc)
            sleeper(min(5, max(0, deadline - time.monotonic())))
    raise RuntimeError(last_error)


def smoke(
    compose_file: Path,
    env_file: Path,
    service: str,
    image: Path,
    target: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    if not image.is_file():
        raise RuntimeError(f"smoke image does not exist: {image}")
    remote_image = f"/tmp/ocrkit-rollout-smoke-{os.getpid()}.png"
    run_checked(
        compose_command(compose_file, env_file, "cp", str(image), f"{service}:{remote_image}"),
        label="smoke fixture copy",
        runner=runner,
    )
    try:
        stdout = run_checked(
            compose_command(
                compose_file,
                env_file,
                "exec",
                "-T",
                "-e",
                f"OCRKIT_ROLLOUT_SMOKE_IMAGE={remote_image}",
                service,
                "python",
                "-c",
                SMOKE_SCRIPT,
            ),
            label="OCR smoke test",
            runner=runner,
        )
        response = json_output(stdout, "OCR smoke test")
        if response.get("model_version") != target["model_version"]:
            raise RuntimeError("OCR smoke response model_version does not match the stable channel target")
        if response.get("ok") is not True:
            raise RuntimeError("OCR smoke response did not report ok")
        return response
    finally:
        runner(
            compose_command(
                compose_file,
                env_file,
                "exec",
                "-T",
                service,
                "python",
                "-c",
                f"from pathlib import Path; Path({remote_image!r}).unlink(missing_ok=True)",
            ),
            text=True,
            capture_output=True,
            check=False,
        )


def rollout(
    compose_file: Path,
    env_file: Path,
    service: str,
    image: Path,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    target_before = read_target(compose_file, env_file, service, runner)
    run_checked(
        compose_command(compose_file, env_file, "up", "-d", "--no-build", "--force-recreate", service),
        label="OCRKit container recreation",
        runner=runner,
    )
    health = wait_for_health(compose_file, env_file, service, target_before, timeout_seconds, runner)
    target_after = read_target(compose_file, env_file, service, runner)
    if target_after != target_before:
        raise RuntimeError("stable channel target changed during rollout")
    smoke_response = smoke(compose_file, env_file, service, image, target_before, runner)
    return {
        "model_version": target_before["model_version"],
        "manifest_key": target_before["manifest_key"],
        "manifest_sha256": target_before["manifest_sha256"],
        "health": health,
        "smoke": {"ok": smoke_response.get("ok"), "model_version": smoke_response.get("model_version")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate OCRKit from the stable model channel and verify the loaded model.")
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose.production.yml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--service", default="ocrkit")
    parser.add_argument("--smoke-image", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if not args.compose_file.is_file():
        raise SystemExit(f"compose file does not exist: {args.compose_file}")
    if not args.env_file.is_file():
        raise SystemExit(f"deployment env file does not exist: {args.env_file}")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    try:
        result = rollout(args.compose_file, args.env_file, args.service, args.smoke_image, args.timeout_seconds)
    except RuntimeError as exc:
        raise SystemExit(f"OCRKit model rollout failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
