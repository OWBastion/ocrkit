from __future__ import annotations

from app.schemas.response import ChallengeData

from .center_summary import CenterSummary
from .left_panel import LeftPanel
from .right_panel import RightPanel


def merge_result(center: CenterSummary, left: LeftPanel, right: RightPanel) -> ChallengeData:
    duration_text = center.duration_text or left.clear_time
    duration_seconds = center.duration_seconds or left.clear_time_seconds

    return ChallengeData(
        challenge_completed=center.completed if center.completed else left.challenge_completed,
        heroes_completed=left.heroes_completed,
        heroes_total=left.heroes_total,
        player=center.player,
        deaths=center.deaths if center.deaths is not None else left.total_deaths,
        skips=center.skips if center.skips is not None else left.total_skips,
        duration_text=duration_text,
        duration_seconds=duration_seconds,
        map_name=right.map_name,
        difficulty=right.difficulty,
        version=right.version,
    )
