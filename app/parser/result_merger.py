from __future__ import annotations

from app.schemas.response import ChallengeData

from .bottom_left_hero import BottomLeftHero
from .center_summary import CenterSummary
from .left_panel import LeftPanel
from .right_panel import RightPanel


def merge_result(
    center: CenterSummary,
    left: LeftPanel,
    bottom_left: BottomLeftHero,
    right: RightPanel,
) -> ChallengeData:
    duration_text = left.clear_time if left.clear_time_seconds is not None else center.duration_text
    duration_seconds = left.clear_time_seconds if left.clear_time_seconds is not None else center.duration_seconds
    return ChallengeData(
        challenge_completed=left.challenge_completed if left.challenge_completed is not None else center.completed,
        heroes_completed=left.heroes_completed,
        heroes_total=left.heroes_total,
        viewer_player=bottom_left.player,
        achievement_title=left.achievement_title,
        achievement_titles=list(left.achievement_titles),
        achievement_unlocked=left.achievement_unlocked,
        deaths=left.total_deaths if left.total_deaths is not None else center.deaths,
        skips=left.total_skips if left.total_skips is not None else center.skips,
        duration_text=duration_text,
        duration_seconds=duration_seconds,
        map_name=right.map_name,
        difficulty=right.difficulty,
        version=right.version,
    )
