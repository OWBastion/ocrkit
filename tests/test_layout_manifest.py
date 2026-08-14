from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.roi_config import load_roi_config
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


@pytest.mark.parametrize("name", ["roi_1280x720.yaml", "roi_1280x800.yaml"])
def test_left_hud_rois_keep_data_and_achievement_regions_separate(name: str) -> None:
    config = load_roi_config(Path(__file__).parents[1] / "configs" / name)
    left = config.rois["left_panel"]
    run_code = config.rois["run_code_panel"]
    achievement = config.rois["achievement_panel"]

    assert run_code.y1 >= left.y1
    assert run_code.y2 <= left.y2
    assert achievement.y1 >= left.y2
    assert achievement.y1 >= run_code.y2
