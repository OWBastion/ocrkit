from __future__ import annotations

from pathlib import Path

import yaml


def test_production_compose_uses_private_service_and_tunnel() -> None:
    compose = yaml.safe_load(Path("docker-compose.production.yml").read_text(encoding="utf-8"))
    ocrkit = compose["services"]["ocrkit"]
    cloudflared = compose["services"]["cloudflared"]

    assert "ports" not in ocrkit
    assert str(ocrkit["image"]).startswith("${OCRKIT_IMAGE:?")
    assert ocrkit["environment"]["OCRKIT_ALLOW_DEBUG"] == "false"
    assert cloudflared["depends_on"]["ocrkit"]["condition"] == "service_healthy"
