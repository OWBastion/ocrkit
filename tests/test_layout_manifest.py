from __future__ import annotations

import json
from pathlib import Path

from scripts.export_layout_manifest import build_manifest, render


def test_checked_in_layout_manifest_matches_authoritative_yaml() -> None:
    root = Path(__file__).parents[1]
    expected = render(build_manifest(root / "configs/roi_1280x720.yaml"))
    actual = (root / "configs/roi_1280x720.manifest.json").read_text(encoding="utf-8")

    assert json.loads(actual) == json.loads(expected)


def test_checked_in_16_10_layout_manifest_matches_authoritative_yaml() -> None:
    root = Path(__file__).parents[1]
    expected = render(build_manifest(root / "configs/roi_1280x800.yaml"))
    actual = (root / "configs/roi_1280x800.manifest.json").read_text(encoding="utf-8")

    assert json.loads(actual) == json.loads(expected)
