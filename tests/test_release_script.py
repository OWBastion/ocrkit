from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training/release_rec_model.sh"
ARTIFACTS_DIR = ROOT / "training/.work/artifacts"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OCRKIT_RELEASE_DET_MODEL", None)
    environment.pop("OCRKIT_MODEL_R2_BUCKET", None)
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _artifact_entries() -> set[Path]:
    if not ARTIFACTS_DIR.exists():
        return set()
    return set(ARTIFACTS_DIR.iterdir())


def test_release_requires_exactly_one_version_argument() -> None:
    before = _artifact_entries()

    result = _run()

    assert result.returncode != 0
    assert "usage:" in result.stderr
    assert _artifact_entries() == before


def test_release_rejects_invalid_version_before_creating_artifact() -> None:
    before = _artifact_entries()

    result = _run("invalid/version")

    assert result.returncode != 0
    assert "invalid release version" in result.stderr
    assert _artifact_entries() == before


def test_release_requires_fixed_detection_model_before_creating_artifact() -> None:
    version = f"pytest-missing-det-{os.getpid()}"
    candidate = ARTIFACTS_DIR / version
    assert not candidate.exists()

    result = _run(version)

    assert result.returncode != 0
    assert "OCRKIT_RELEASE_DET_MODEL" in result.stderr
    assert not candidate.exists()
