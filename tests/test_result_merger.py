from app.parser.center_summary import CenterSummary
from app.parser.left_panel import LeftPanel
from app.parser.result_merger import merge_result
from app.parser.right_panel import RightPanel


def test_merge_result_prefers_left_deaths_skips() -> None:
    center = CenterSummary(
        completed=True,
        player="player",
        deaths=94,
        skips=0,
        duration_text="2小时20分38秒",
        duration_seconds=8438.0,
    )
    left = LeftPanel(
        heroes_completed=51,
        heroes_total=51,
        challenge_completed=True,
        total_deaths=114,
        total_skips=0,
        clear_time="2小时20分38秒",
        clear_time_seconds=8438.0,
    )
    right = RightPanel(map_name="萨摩亚：地狱", difficulty="地狱", version="26.0513.6")

    out = merge_result(center, left, right)
    assert out.deaths == 114
    assert out.skips == 0


def test_merge_result_uses_center_deaths_skips_without_left_values() -> None:
    center = CenterSummary(
        completed=True,
        player="player",
        deaths=94,
        skips=0,
        duration_text="2小时20分38秒",
        duration_seconds=8438.0,
    )
    left = LeftPanel(
        heroes_completed=51,
        heroes_total=51,
        challenge_completed=True,
        total_deaths=None,
        total_skips=None,
        clear_time="2小时20分38秒",
        clear_time_seconds=8438.0,
    )
    right = RightPanel(map_name="萨摩亚：地狱", difficulty="地狱", version="26.0513.6")

    out = merge_result(center, left, right)
    assert out.deaths == 94
    assert out.skips == 0
