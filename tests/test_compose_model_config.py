from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_passes_model_r2_configuration_without_secrets() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["ocrkit"]["environment"]

    expected = {
        "OCRKIT_R2_ENDPOINT_URL",
        "OCRKIT_R2_ACCESS_KEY_ID",
        "OCRKIT_R2_SECRET_ACCESS_KEY",
        "OCRKIT_API_TOKEN",
        "OCRKIT_ALLOW_DEBUG",
        "OCRKIT_R2_REGION_NAME",
        "OCRKIT_R2_DEFAULT_BUCKET",
        "OCRKIT_R2_ALLOWED_BUCKETS",
        "OCRKIT_MODEL_MANIFEST_KEY",
        "OCRKIT_MODEL_RELEASE_CHANNEL_KEY",
        "OCRKIT_MODEL_REFRESH_SECONDS",
        "OCRKIT_MODEL_CACHE_DIR",
    }
    assert expected <= set(environment)
    assert all(environment[name] != "" for name in expected)
    for name in {"OCRKIT_R2_ACCESS_KEY_ID", "OCRKIT_R2_SECRET_ACCESS_KEY"}:
        assert str(environment[name]).startswith("${")
    assert str(environment["OCRKIT_API_TOKEN"]).startswith("${")
    assert environment["OCRKIT_ALLOW_DEBUG"] == "${OCRKIT_ALLOW_DEBUG:-false}"


def test_compose_persists_model_cache_in_named_volume() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    assert "ocrkit-models" in compose["volumes"]
    assert "ocrkit-models:/var/lib/ocrkit/models" in compose["services"]["ocrkit"]["volumes"]
