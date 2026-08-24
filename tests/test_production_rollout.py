from __future__ import annotations

from pathlib import Path

import pytest

from training.scripts.production_rollout import compose_command, validate_health, validate_target


def test_target_requires_immutable_manifest_identity() -> None:
    target = validate_target(
        {
            "channel_key": "models/pp-ocrv6-small/channels/stable.json",
            "manifest_key": "models/pp-ocrv6-small/2026.08.24/manifest.json",
            "model_version": "2026.08.24",
            "manifest_sha256": "a" * 64,
        }
    )

    assert target["model_version"] == "2026.08.24"

    with pytest.raises(RuntimeError, match="outside"):
        validate_target({**target, "manifest_key": "other/model/manifest.json"})


def test_health_must_match_stable_target() -> None:
    target = {
        "channel_key": "models/pp-ocrv6-small/channels/stable.json",
        "manifest_key": "models/pp-ocrv6-small/2026.08.24/manifest.json",
        "model_version": "2026.08.24",
        "manifest_sha256": "a" * 64,
    }
    validate_health({"ok": True, "model_version": "2026.08.24"}, target)

    with pytest.raises(RuntimeError, match="does not match"):
        validate_health({"ok": True, "model_version": "old"}, target)


def test_compose_rollout_recreates_without_rebuilding() -> None:
    command = compose_command(Path("docker-compose.production.yml"), Path(".env"), "up", "-d", "--no-build", "--force-recreate", "ocrkit")

    assert command[-5:] == ["up", "-d", "--no-build", "--force-recreate", "ocrkit"]
    assert "--build" not in command
