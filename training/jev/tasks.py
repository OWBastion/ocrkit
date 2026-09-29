"""Bounded pre-review task construction for Jev-Omni decisions.

Per #25 scope: each Studio review row becomes one typed-choice decision whose
options are the de-duplicated OCR candidate texts plus ``OPTION_NONE_CORRECT``
(the crop is valid but none of the candidates is exact) and
``OPTION_NOT_VALID`` (the crop holds no valid target text for the ROI). The
prompt text is versioned because it is part of the recorded decision input.
"""

from __future__ import annotations

from .records import PreReviewRecord

QUESTION = "Which option matches the visible text in the image exactly?"

# Expected-content hints per ROI family (prompt v2). These describe what the
# field is supposed to contain so the model can use the not-valid option
# meaningfully instead of only matching text shape.
ROI_HINTS = {
    "left_panel": (
        " It should contain a challenge progress line such as hero counts "
        "('英雄: 3/15', '下一个英雄'), death/skip totals ('总计阵亡/跳过', '97/0'), "
        "clear time ('1小时57分49秒'), or an achievement title."
    ),
    "right_panel": (
        " It should contain a server or map info line such as map name plus "
        "difficulty ('66号公路：地狱'), performance stats ('FPS: 173', "
        "'VRM: 5306 MB', 'PING: 22 MS', '服务器负载47'), or the game version."
    ),
    "center_banner": " It should contain the complete centered banner message.",
    "bottom_left_hero": " It should contain the bottom-left hero/player identifier or score values.",
    "run_code_panel": (
        " The expected content is the match run code: three groups of four digits "
        "separated by hyphens, optionally after a '本局代码' or 'Run Code' label."
    ),
    "run_code_right_panel": (
        " The expected content is the match run code: three groups of four digits "
        "separated by hyphens, optionally after a '本局代码' or 'Run Code' label."
    ),
    "achievement_panel": " It should contain an achievement name or a completion mark ('✓').",
}


def build_state(record: PreReviewRecord) -> str:
    state = (
        "The image is a small cropped text region from the "
        f"'{record.roi}' field of a video-game match results screen."
    )
    if record.prompt_version != "jev-prereview-v1":
        state += ROI_HINTS.get(record.roi, "")
    elif record.roi in {"run_code_panel", "run_code_right_panel"}:
        state += ROI_HINTS[record.roi]
    return state


def build_task(record: PreReviewRecord, image_path: str) -> dict[str, object]:
    """The exact worker request for one record; mirrors the digest input."""
    return {
        "task_id": record.record_id,
        "image": image_path,
        "state": build_state(record),
        "question": QUESTION,
        "options": list(record.options),
    }
