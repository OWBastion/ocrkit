from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.roi_config import load_roi_config
from app.ocr.engine import OcrChunk, OcrResult
from app.parser.ai_mark import is_ai_mark
from app.parser.event_stack import has_check_mark, parse_event_stack
from app.service import extract_structured

SCALE = 3.0


def _stack_image(check_rows: tuple[int, ...]) -> np.ndarray:
    """ROI-sized dark image with a bright check icon right of each requested line (ROI coordinates)."""
    gray = np.full((187, 280), 30, dtype=np.uint8)
    for y in check_rows:
        gray[y + 2:y + 12, 115:125] = 255
    return gray


def _chunk(text: str, y: int, x2: int = 110) -> OcrChunk:
    return OcrChunk(text=text, score=0.95, box=(33 * SCALE, y * SCALE, x2 * SCALE, (y + 14) * SCALE))


def test_check_icon_is_found_only_after_checked_lines() -> None:
    gray = _stack_image((50,))
    assert has_check_mark(gray, (33, 50, 110, 64))
    assert not has_check_mark(gray, (33, 20, 110, 34))


def test_bright_wall_behind_the_line_is_not_a_check_icon() -> None:
    assert not has_check_mark(np.full((187, 280), 255, dtype=np.uint8), (33, 50, 110, 64))


def test_event_with_numbers_then_achievements_without_ocr_readable_marks() -> None:
    chunks = [
        _chunk("心之钢(27秒)", 3),
        _chunk("阵亡后获得1层", 19),
        _chunk("每层永久提供0.75%最大生命值", 35, x2=198),
        _chunk("生命守护生命", 60),
        _chunk("V我50", 76),
    ]
    stack = parse_event_stack(chunks, _stack_image((60, 76)), SCALE)

    assert stack.achievements == ("生命守护生命", "V我50")
    assert stack.event is not None
    assert (stack.event.name, stack.event.duration_seconds) == ("心之钢", 27)
    assert stack.event.description == ["阵亡后获得1层", "每层永久提供0.75%最大生命值"]
    assert [(n.text, n.value, n.unit) for n in stack.event.numbers] == [("1层", 1.0, "层"), ("0.75%", 0.75, "%")]


def test_achievements_move_up_when_there_is_no_event_and_are_capped_at_five() -> None:
    chunks = [_chunk(f"成就{i}", 3 + i * 16) for i in range(7)]
    stack = parse_event_stack(chunks, _stack_image(tuple(3 + i * 16 for i in range(7))), SCALE)

    assert stack.event is None
    assert stack.achievements == tuple(f"成就{i}" for i in range(5))


def test_text_marker_still_counts_without_an_icon() -> None:
    stack = parse_event_stack([_chunk("新称号 ✓", 3)], np.zeros((187, 280), np.uint8), SCALE)
    assert stack.achievements == ("新称号",)


def test_ai_mark_text() -> None:
    assert is_ai_mark("豆包AI生成")
    assert is_ai_mark("AI 生成")
    assert not is_ai_mark("AI")
    assert not is_ai_mark("")


def test_service_returns_event_achievements_and_ai_mark_as_additive_fields() -> None:
    root = Path(__file__).parents[1]
    config = load_roi_config(root / "configs/roi_1280x720.yaml")
    image = np.full((720, 1280, 3), 30, dtype=np.uint8)
    image[228 + 62:228 + 72, 20 + 115:20 + 125] = 255  # check icon after the achievement line at ROI y=60

    class Engine:
        def recognize(self, processed: np.ndarray) -> OcrResult:
            if processed.shape[:2] == (561, 840):
                chunks = [_chunk("心之钢(27秒)", 3), _chunk("阵亡后获得1层", 19), _chunk("生命守护生命", 60)]
                return OcrResult(" ".join(c.text for c in chunks), 0.95, chunks)
            if processed.shape[:2] == (42, 140):
                return OcrResult("豆包AI生成", 0.98, [])
            return OcrResult("", 0.0, [])

    response = extract_structured(image, config, [], {}, Engine(), False, "r1", "test", "test", config.version)

    assert response.data is not None
    assert response.data.achievement_titles == ["生命守护生命"]
    assert response.data.event is not None and response.data.event.name == "心之钢"
    assert response.data.ai_mark_detected is True
    assert response.fields["event"].source_roi == ["achievement_panel"]
    assert response.fields["ai_mark_detected"].value is True


def test_service_reads_the_real_player_name_above_the_server_load() -> None:
    root = Path(__file__).parents[1]
    config = load_roi_config(root / "configs/roi_1280x720.yaml")

    class Engine:
        def __init__(self, text: str) -> None:
            self.text = text

        def recognize(self, processed: np.ndarray) -> OcrResult:
            return OcrResult(self.text, 0.93, []) if processed.shape[:2] == (36, 235) else OcrResult("", 0.0, [])

    image = np.full((720, 1280, 3), 30, dtype=np.uint8)
    named = extract_structured(image, config, [], {}, Engine("云雀 服务器负载47"), False, "r1", "test", "test", config.version)
    unnamed = extract_structured(image, config, [], {}, Engine("服务器负载62"), False, "r2", "test", "test", config.version)

    assert named.data is not None and (named.data.real_player, named.data.server_load) == ("云雀", 47)
    assert named.fields["real_player"].source_roi == ["server_load_panel"]
    assert named.fields["real_player"].confidence == 0.93
    assert unnamed.data is not None and (unnamed.data.real_player, unnamed.data.server_load) == (None, 62)
    assert unnamed.fields["real_player"].status == "missing"


def test_right_panel_status_lines() -> None:
    from app.parser.right_panel import parse_right_panel

    text = "里阿尔托：地狱 [2026镜中回响] 距自动重开46分51秒 已运行1小时23分8秒 版本26.1002.2 8385-4192-4002 服务器负载107"
    panel = parse_right_panel(text, ["里阿尔托"])

    assert panel.mode == "2026镜中回响"
    assert panel.restart_in_seconds == 46 * 60 + 51
    assert panel.uptime_seconds == 3600 + 23 * 60 + 8
    assert panel.server_load == 107


def test_garbled_duration_is_not_guessed() -> None:
    from app.parser.right_panel import parse_right_panel

    panel = parse_right_panel("[随机事件5.0] 距自动重开1小时分37秒 已运行3小21分22秒 服务器负载", [])

    assert panel.mode == "随机事件5.0"
    assert panel.restart_in_seconds is None
    assert panel.uptime_seconds is None
    assert panel.server_load is None


def test_mode_ignores_spacing_the_hud_and_ocr_disagree_about() -> None:
    from app.parser.right_panel import parse_right_panel

    assert parse_right_panel("[2026 镜中回响]", []).mode == "2026镜中回响"
    assert parse_right_panel("[随机事件 5.0]", []).mode == "随机事件5.0"


def test_text_elsewhere_in_the_tall_region_is_not_an_achievement() -> None:
    # From a recent screenshot: a world sign at the far left, a name plate on the right, with a bright letter
    # after the sign text that looks like a check icon.
    gray = np.full((187, 280), 30, dtype=np.uint8)
    gray[163:173, 24:34] = 255
    chunks = [
        OcrChunk("TERI", 0.9, (0 * SCALE, 161 * SCALE, 21 * SCALE, 175 * SCALE)),
        OcrChunk("好男人从不", 0.9, (226 * SCALE, 108 * SCALE, 280 * SCALE, 122 * SCALE)),
    ]
    stack = parse_event_stack(chunks, gray, SCALE)

    assert stack.achievements == () and stack.event is None


def test_the_stack_must_start_at_the_top_and_have_no_gaps() -> None:
    chunks = [_chunk("成就甲", 6), _chunk("成就乙", 22), _chunk("远处的文字", 120)]
    stack = parse_event_stack(chunks, _stack_image((6, 22, 120)), SCALE)

    assert stack.achievements == ("成就甲", "成就乙")
    assert parse_event_stack([_chunk("太低了", 90)], _stack_image((90,)), SCALE).achievements == ()


def test_achievements_centred_under_a_wide_event_are_kept() -> None:
    def centred(text: str, y: int) -> OcrChunk:
        return OcrChunk(text=text, score=0.95, box=(100 * SCALE, y * SCALE, 150 * SCALE, (y + 14) * SCALE))

    chunks = [_chunk("作弊：我要验牌(10秒)", 3, x2=190), _chunk("下个事件必为增益", 35), centred("生命守护生命", 66), centred("把他们上市", 82)]
    stack = parse_event_stack(chunks, _stack_image((66, 82)) * 0 + _centred_icons(), SCALE)

    assert stack.achievements == ("生命守护生命", "把他们上市")
    assert stack.event is not None and stack.event.name == "作弊：我要验牌"


def _centred_icons() -> np.ndarray:
    gray = np.full((187, 280), 30, dtype=np.uint8)
    for y in (66, 82):
        gray[y + 2:y + 12, 165:175] = 255
    return gray
