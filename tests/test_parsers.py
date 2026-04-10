from ocrkit.parsers import match_map_label, parse_difficulty, parse_run_from_texts, parse_time_to_seconds


def test_parse_time_chinese_hms() -> None:
    assert parse_time_to_seconds("2小时27分39秒") == 8859.0


def test_parse_time_mmss() -> None:
    assert parse_time_to_seconds("12:30.5") == 750.5


def test_parse_difficulty() -> None:
    assert parse_difficulty("当前难度地狱无法跳过英雄") == "地狱"


def test_map_match_contains() -> None:
    label, score = match_map_label(["欢迎来到三合一大地图 伊利奥斯"], ["伊利奥斯", "尼泊尔"])
    assert label == "伊利奥斯"
    assert score == 1.0


def test_parse_run_banner() -> None:
    parsed = parse_run_from_texts(
        center_banner="祝贺 好男人从不过夜 以 115 次阵亡 & 0 次跳过 耗时 2小时27分39秒 通关",
        top_bar="当前难度地狱无法跳过英雄",
        left_panel="通关总计耗时 2小时27分39秒",
        map_labels=["伊利奥斯", "尼泊尔"],
    )
    assert parsed.passed is True
    assert parsed.player_name == "好男人从不过夜"
    assert parsed.deaths == 115
    assert parsed.skips == 0
    assert parsed.time_sec == 8859.0
    assert parsed.difficulty == "地狱"
