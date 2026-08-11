import json
from pathlib import Path


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
