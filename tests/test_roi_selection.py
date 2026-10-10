from pathlib import Path

import numpy as np

from app.core.roi_config import RoiBox, RoiConfig, load_roi_config
from app.image.roi import anchored_rois, crop_all_rois, select_roi_config


def _config(width: int, height: int, version: str) -> RoiConfig:
    return RoiConfig(width, height, {"left_panel": RoiBox(0, 0, 1, 1)}, version)


def test_select_roi_config_uses_matching_source_aspect_ratio() -> None:
    widescreen = _config(1280, 720, "1280x720-v6")
    tall = _config(1280, 800, "1280x800-v1")

    assert select_roi_config(np.zeros((1600, 2560, 3), dtype=np.uint8), (widescreen, tall)) == tall
    assert select_roi_config(np.zeros((1440, 2560, 3), dtype=np.uint8), (widescreen, tall)) == widescreen


def test_ultrawide_keeps_height_scale_and_anchors_hud_to_screen_edges() -> None:
    config = load_roi_config(Path("configs/roi_1280x720.yaml"))
    rng = np.random.default_rng(7)
    standard = rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8)
    extra = rng.integers(0, 256, (1440, 880, 3), dtype=np.uint8)
    ultrawide = np.concatenate([standard[:, :1280], extra, standard[:, 1280:]], axis=1)

    _, standard_crops = crop_all_rois(standard, config)
    normalized, wide_crops = crop_all_rois(ultrawide, config)

    assert normalized.shape[:2] == (720, 1720)
    for name in ("left_panel", "run_code_panel", "achievement_panel", "bottom_left_hero", "right_panel", "run_code_right_panel"):
        assert np.array_equal(wide_crops[name], standard_crops[name]), name
    assert anchored_rois(config, 1720)["right_panel"].x2 == config.rois["right_panel"].x2 + 440
