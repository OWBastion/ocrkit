from app.parser.bottom_left_hero import parse_bottom_left_hero


def test_parse_bottom_left_hero_player() -> None:
    out = parse_bottom_left_hero("385 385 N! 3 训犬大师")
    assert out.player == "训犬大师"
    assert out.achievement_title == "训犬大师"


def test_parse_bottom_left_hero_without_title() -> None:
    out = parse_bottom_left_hero("")
    assert out.achievement_title is None
