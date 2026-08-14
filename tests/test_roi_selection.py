import numpy as np

from app.core.roi_config import RoiBox, RoiConfig
from app.image.roi import select_roi_config


def _config(width: int, height: int, version: str) -> RoiConfig:
    return RoiConfig(width, height, {"left_panel": RoiBox(0, 0, 1, 1)}, version)


def test_select_roi_config_uses_matching_source_aspect_ratio() -> None:
    widescreen = _config(1280, 720, "1280x720-v6")
    tall = _config(1280, 800, "1280x800-v1")

    assert select_roi_config(np.zeros((1600, 2560, 3), dtype=np.uint8), (widescreen, tall)) == tall
    assert select_roi_config(np.zeros((1440, 2560, 3), dtype=np.uint8), (widescreen, tall)) == widescreen
