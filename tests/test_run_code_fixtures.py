from pathlib import Path

import cv2
import numpy as np

from app.core.roi_config import load_roi_config
from app.image.roi import normalize_canvas


def test_clean_settlement_fixture_code_line_is_inside_the_16_9_run_code_roi() -> None:
    root = Path(__file__).parents[1]
    image = cv2.imread(str(root / "tests/fixtures/run_code/run_code_clean_1280.png"))
    assert image is not None

    normalized = normalize_canvas(image, 1280, 720)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    visible_rows = np.flatnonzero((gray[:, :400] > 100).any(axis=1))
    code_rows = visible_rows[visible_rows > 220]
    code_xs = np.flatnonzero((gray[code_rows, :] > 100).any(axis=0))
    box = load_roi_config(root / "configs/roi_1280x720.yaml").rois["run_code_panel"]

    assert box.x1 <= int(code_xs.min())
    assert int(code_xs.max()) < box.x2
    assert box.y1 <= int(code_rows.min())
    assert int(code_rows.max()) < box.y2
