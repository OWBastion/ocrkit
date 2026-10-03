from app.parser.bottom_left_hero import parse_bottom_left_hero


def test_parse_bottom_left_hero_player() -> None:
    out = parse_bottom_left_hero("385 385 N! 3 训犬大师")
    assert out.player == "训犬大师"


def _chunk(text: str, x1: int, y1: int, x2: int, y2: int):
    from app.ocr.engine import OcrChunk

    return OcrChunk(text, 0.95, (x1, y1, x2, y2))


def test_name_is_the_widest_fragment_not_the_last_one() -> None:
    # Measured on fixtures: the level badge and rank icon are read after the name.
    chunks = [_chunk("712712", 213, 31, 378, 84), _chunk("训犬大师", 202, 134, 308, 175), _chunk("0", 173, 151, 189, 174)]
    assert parse_bottom_left_hero("712712 训犬大师 0", chunks).player == "训犬大师"

    chunks = [_chunk("406|406", 226, 26, 402, 77), _chunk("初一", 253, 147, 308, 183), _chunk("UI", 222, 161, 237, 176)]
    assert parse_bottom_left_hero("406|406 初一 UI", chunks).player == "初一"


def test_level_badge_digit_glued_to_the_name_is_removed() -> None:
    assert parse_bottom_left_hero("847|847 3初一", [_chunk("847|847", 224, 25, 393, 79), _chunk("3初一", 164, 138, 262, 184)]).player == "初一"
    assert parse_bottom_left_hero("1ERUILUVATAR", [_chunk("1ERUILUVATAR", 166, 197, 374, 252)]).player == "ERUILUVATAR"
    # Digits inside or after a name stay.
    assert parse_bottom_left_hero("路人2号", [_chunk("路人2号", 0, 0, 90, 20)]).player == "路人2号"


def test_a_name_made_only_of_digits_is_still_found_when_it_is_the_lowest_fragment() -> None:
    chunks = [_chunk("500|500", 200, 20, 400, 60), _chunk("12345", 200, 140, 300, 175)]
    assert parse_bottom_left_hero("500|500 12345", chunks).player == "12345"


def test_level_badge_digit_separated_from_the_name_by_a_space_is_removed() -> None:
    assert parse_bottom_left_hero("3 训犬大师", [_chunk("3 训犬大师", 164, 138, 308, 184)]).player == "训犬大师"
