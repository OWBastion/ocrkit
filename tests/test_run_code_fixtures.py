import json
from pathlib import Path

import cv2
import numpy as np

from app.core.roi_config import load_roi_config
from app.image.roi import normalize_canvas


def test_run_code_fixture_set_covers_supported_and_conservative_paths() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "run_code"
    cases = json.loads((fixture_dir / "cases.json").read_text(encoding="utf-8"))
    cases_by_id = {case["id"]: case for case in cases}

    assert set(cases_by_id) == {
        "clean_1280",
        "high_res_2560",
        "compressed",
        "scaled_1600",
        "missing",
        "cropped",
        "ambiguous",
        "malformed",
    }
    assert cases_by_id["clean_1280"]["expected"]["run_code"] == "4821-7354-1926"
    assert cases_by_id["high_res_2560"]["expected"]["run_code"] == "7246-3815-9472"
    assert all((fixture_dir / case["image"]).is_file() for case in cases)


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
