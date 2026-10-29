from __future__ import annotations

from app.schemas.response import ChallengeData, EventData

from .bottom_left_hero import BottomLeftHero
from .center_summary import CenterSummary
from .left_panel import LeftPanel
from .right_panel import RightPanel
from .settlement_card import SettlementCard


def _first(*values):
    return next((value for value in values if value is not None), None)


def merge_result(
    center: CenterSummary,
    left: LeftPanel,
    bottom_left: BottomLeftHero,
    right: RightPanel,
    achievement_panel_text: str | None = None,
    run_code: str | None = None,
    event: EventData | None = None,
    ai_mark_detected: bool | None = None,
    card: SettlementCard | None = None,
) -> ChallengeData:
    card = card or SettlementCard()
    if left.clear_time_seconds is not None:
        duration_text, duration_seconds = left.clear_time, left.clear_time_seconds
    elif card.duration_seconds is not None:
        duration_text, duration_seconds = card.duration_text, card.duration_seconds
    else:
        duration_text, duration_seconds = center.duration_text, center.duration_seconds
    return ChallengeData(
        challenge_completed=left.challenge_completed if left.challenge_completed is not None else center.completed or card.completed,
        heroes_completed=left.heroes_completed,
        heroes_total=left.heroes_total,
        viewer_player=bottom_left.player,
        achievement_title=left.achievement_title,
        achievement_titles=list(left.achievement_titles),
        achievement_unlocked=left.achievement_unlocked,
        achievement_panel_text=achievement_panel_text,
        deaths=_first(left.total_deaths, card.deaths, center.deaths),
        skips=_first(left.total_skips, card.skips, center.skips),
        duration_text=duration_text,
        duration_seconds=duration_seconds,
        map_name=_first(right.map_name, card.map_name),
        map_variant=_first(right.map_variant, card.map_variant),
        difficulty=_first(right.difficulty, card.difficulty),
        version=_first(right.version, card.version),
        run_code=run_code,
        event=event,
        ai_mark_detected=ai_mark_detected,
        mode=right.mode,
        restart_in_seconds=right.restart_in_seconds,
        uptime_seconds=right.uptime_seconds,
        server_load=right.server_load,
        real_player=right.real_player,
    )
