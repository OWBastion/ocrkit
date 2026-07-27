from app.parser.center_summary import CenterSummary
from app.parser.bottom_left_hero import BottomLeftHero
from app.parser.left_panel import LeftPanel
from app.parser.result_merger import merge_result
from app.parser.right_panel import RightPanel


def test_merge_result_prefers_left_deaths_skips() -> None:
    center = CenterSummary(
        completed=True,
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

    out = merge_result(center, left, BottomLeftHero(player="bottom-player"), right)
    assert out.deaths == 114
    assert out.skips == 0
    assert out.viewer_player == "bottom-player"
    assert out.achievement_title is None
    assert out.achievement_unlocked is None


def test_merge_result_prefers_left_duration() -> None:
    center = CenterSummary(
        completed=True,
        deaths=94,
        skips=0,
        duration_text="3小时18 12秒",
        duration_seconds=10800.0,
    )
    left = LeftPanel(
        heroes_completed=51,
        heroes_total=51,
        challenge_completed=True,
        total_deaths=94,
        total_skips=0,
        clear_time="3小时18分12秒",
        clear_time_seconds=11892.0,
    )
    right = RightPanel(map_name="中城", difficulty="地狱", version="26.0613.3")

    out = merge_result(center, left, BottomLeftHero(player=None), right)
    assert out.duration_text == "3小时18分12秒"
    assert out.duration_seconds == 11892.0


def test_merge_result_uses_center_deaths_skips_without_left_values() -> None:
    center = CenterSummary(
        completed=True,
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

    out = merge_result(center, left, BottomLeftHero(player=None), right)
    assert out.deaths == 94
    assert out.skips == 0


def test_merge_result_uses_center_completion_without_left_value() -> None:
    center = CenterSummary(
        completed=True,
        deaths=None,
        skips=None,
        duration_text=None,
        duration_seconds=None,
    )
    left = LeftPanel(
        heroes_completed=None,
        heroes_total=None,
        challenge_completed=None,
        total_deaths=None,
        total_skips=None,
        clear_time=None,
        clear_time_seconds=None,
    )
    right = RightPanel(map_name=None, difficulty=None, version=None)

    out = merge_result(center, left, BottomLeftHero(player=None), right)
    assert out.challenge_completed is True


def test_merge_result_uses_center_duration_when_left_time_is_invalid() -> None:
    center = CenterSummary(
        completed=True,
        deaths=None,
        skips=None,
        duration_text="1小时26分36秒",
        duration_seconds=5196.0,
    )
    left = LeftPanel(
        heroes_completed=None,
        heroes_total=None,
        challenge_completed=None,
        total_deaths=None,
        total_skips=None,
        clear_time="1",
        clear_time_seconds=None,
    )
    right = RightPanel(map_name=None, difficulty=None, version=None)

    out = merge_result(center, left, BottomLeftHero(player=None), right)
    assert out.duration_text == "1小时26分36秒"
    assert out.duration_seconds == 5196.0


def test_merge_result_uses_viewer_player_only() -> None:
    center = CenterSummary(
        completed=True,
        deaths=None,
        skips=None,
        duration_text=None,
        duration_seconds=None,
    )
    left = LeftPanel(None, None, None, None, None, None, None)
    right = RightPanel(map_name=None, difficulty=None, version=None)

    out = merge_result(center, left, BottomLeftHero(player="订犬大师"), right)
    assert out.viewer_player == "订犬大师"
    assert out.achievement_unlocked is None


def test_merge_result_uses_viewer_player_without_center_player() -> None:
    center = CenterSummary(True, None, None, None, None)
    left = LeftPanel(None, None, None, None, None, None, None)
    right = RightPanel(None, None, None)

    out = merge_result(center, left, BottomLeftHero(player="训犬大师"), right)
    assert out.viewer_player == "训犬大师"
