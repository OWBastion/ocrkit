from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path("training/scripts/build_manifest.py")
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(monkeypatch: pytest.MonkeyPatch, artifact_dir: Path, version: str) -> None:
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["build_manifest.py", "--artifact-dir", str(artifact_dir), "--version", version])
    module.main()


def test_build_manifest_rejects_invalid_release_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    with pytest.raises(SystemExit, match="version may contain"):
        _run_main(monkeypatch, artifact_dir, "2026/07/12")


def test_build_manifest_rejects_incomplete_artifact_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "det.onnx").write_bytes(b"det")

    with pytest.raises(SystemExit, match="required artifact is missing"):
        _run_main(monkeypatch, artifact_dir, "2026.07.12-01")
