from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training/release_rec_model.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in {
        "OCRKIT_R2_ENDPOINT_URL",
        "OCRKIT_R2_ACCESS_KEY_ID",
        "OCRKIT_R2_SECRET_ACCESS_KEY",
        "OCRKIT_R2_DEFAULT_BUCKET",
    }:
        environment[name] = ""
    return subprocess.run(
        [str(SCRIPT), *args], cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )


def test_release_requires_no_positional_arguments() -> None:
    result = _run("2026.07.12-01")

    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_release_loads_required_r2_contract_before_creating_artifact() -> None:
    result = _run()

    assert result.returncode != 0
    assert "OCRKIT_R2_DEFAULT_BUCKET" in result.stderr


def test_release_accepts_an_explicit_studio_checkpoint() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--checkpoint" in text
    assert "publish_model_channel.py" in text
    assert "--release-channel" in text
    assert "candidate.json" in text
    assert "stable channel requires explicit promotion" in text
