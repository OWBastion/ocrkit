from app.parser.run_code import parse_run_code
from app.parser.settlement_card import parse_settlement_card

MAPS = ["66号公路", "漓江塔", "巴黎"]
CARD = "挑战完成·01:02:03 玩家他又·通关代码5681-4690-4470 66号公路·地狱·V26.1027.3 阵亡12·跳过3 称号{titles} 剩余19秒"


def test_parse_settlement_card_reads_every_line() -> None:
    card = parse_settlement_card(CARD.format(titles="—"), MAPS)

    assert card.completed is True
    assert card.duration_text == "01:02:03"
    assert card.duration_seconds == 3723.0
    assert card.player == "他又"
    assert (card.deaths, card.skips) == (12, 3)
    assert (card.map_name, card.difficulty, card.version) == ("66号公路", "地狱", "26.1027.3")
    assert card.titles == []


def test_parse_settlement_card_reads_earned_titles() -> None:
    card = parse_settlement_card(CARD.format(titles="Ура! Ура! · 我大受震撼"), MAPS)

    assert card.titles == ["Ура! Ура!", "我大受震撼"]


def test_parse_settlement_card_treats_misread_dash_as_no_titles() -> None:
    assert parse_settlement_card(CARD.format(titles="一"), MAPS).titles == []


def test_parse_settlement_card_without_title_is_not_a_completion() -> None:
    card = parse_settlement_card("国服代码【SVMQK】亚服【1ACCZ】500强交流群：445152554", MAPS)

    assert card.completed is False
    assert card.duration_seconds is None
    assert card.map_name is None


def test_parse_settlement_card_leaves_unreadable_lines_missing() -> None:
    card = parse_settlement_card("挑战完成·00:10:00 玩家他又", MAPS)

    assert card.completed is True
    assert card.player is None
    assert card.deaths is None and card.skips is None
    assert card.map_name is None and card.difficulty is None and card.version is None


def test_completion_card_run_code_label_is_accepted() -> None:
    parsed = parse_run_code("玩家他又·通关代码5681-4690-4470")

    assert parsed.value == "5681-4690-4470"
    assert parsed.status == "ok"


def test_disabled_notice_on_the_titles_line_is_not_a_title() -> None:
    text = "挑战完成 · 00:00:00 玩家 他又 · 通关代码 5681-4690-4470 66号公路 · 地狱 · v26.1027.3 阵亡 0 · 跳过 0 称号 成就挑战已禁用 剩余 19 秒"
    card = parse_settlement_card(text, ["66号公路"])
    assert card.achievement_challenge_disabled is True
    assert card.titles == []


def test_titles_line_without_the_notice_leaves_it_unknown() -> None:
    text = "挑战完成 · 00:00:00 玩家 他又 · 通关代码 5681-4690-4470 66号公路 · 地狱 · v26.1027.3 阵亡 0 · 跳过 0 称号 — 剩余 19 秒"
    assert parse_settlement_card(text, ["66号公路"]).achievement_challenge_disabled is None
