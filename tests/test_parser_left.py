from app.parser.left_panel import parse_achievement_titles, parse_left_panel


def test_parse_left_panel() -> None:
    text = "英雄: 5I / 5I 挑战完成 总计死亡/跳过 94/O 通关总计时 2小时9分S9秒"
    out = parse_left_panel(text)
    assert out.heroes_completed == 51
    assert out.heroes_total == 51
    assert out.challenge_completed is True
    assert out.total_deaths == 94
    assert out.total_skips == 0
    assert out.clear_time_seconds == 7799.0


def test_parse_left_panel_samoa_variant() -> None:
    text = "英雄：51/51 挑战完成 总计阵亡/跳过 114/0 通关总计耗时 2小时20分38秒"
    out = parse_left_panel(text)
    assert out.heroes_completed == 51
    assert out.heroes_total == 51
    assert out.challenge_completed is True
    assert out.total_deaths == 114
    assert out.total_skips == 0
    assert out.clear_time_seconds == 8438.0


def test_parse_left_panel_deaths_skips_noise() -> None:
    text = "英雄：51 / 51 挑战完成 总计阵亡/跳过\nI14／O 通关总计耗时 2小时20分38秒"
    out = parse_left_panel(text)
    assert out.total_deaths == 114
    assert out.total_skips == 0

def test_parse_left_panel_deaths_skips_guo_variant() -> None:
    text = "英雄：51/51 挑战完成 总计阵亡过 94/0 通关总计耗时 2小时9分59秒"
    out = parse_left_panel(text)
    assert out.total_deaths == 94
    assert out.total_skips == 0


def test_parse_left_panel_time_without_clear_prefix() -> None:
    out = parse_left_panel("英雄:51/51 挑战完成 通美总计耗时 1小时27分45秒")
    assert out.clear_time_seconds == 5265.0


def test_parse_left_panel_does_not_use_boost_stats_for_deaths_skips() -> None:
    out = parse_left_panel("总计阵亡跳过 9210 增益/减益/总计 30/29169")
    assert out.total_deaths is None
    assert out.total_skips is None


def test_parse_left_panel_deaths_skips_label_ocr_variant() -> None:
    out = parse_left_panel("总计车二跳过 123/0 增益/减益/总计 30/29169")
    assert out.total_deaths == 123
    assert out.total_skips == 0


def test_parse_achievement_titles_with_checkmark() -> None:
    assert parse_achievement_titles("钢门 ✓") == ("钢门",)


def test_parse_achievement_titles_with_ocr_checkmark_variant() -> None:
    assert parse_achievement_titles("生命守护生命 L") == ("生命守护生命",)


def test_parse_left_panel_parses_unlabeled_deaths_skips_before_boost_stats() -> None:
    out = parse_left_panel("106.0 增益/减益/总计 33/31/82")
    assert out.total_deaths == 106
    assert out.total_skips == 0


def test_parse_left_panel_parses_variant_time_label() -> None:
    out = parse_left_panel("通关总训耗时 1小时42分15秒")
    assert out.clear_time_seconds == 6135.0


def test_parse_achievement_titles_matches_multiple_titles_in_order() -> None:
    assert parse_achievement_titles("钢门 ✓\n幸运星 ✓\n开了 ✓\nV我50 ✓\n牢大 ✓") == (
        "钢门",
        "幸运星",
        "开了",
        "V我50",
        "牢大",
    )


def test_parse_left_panel_does_not_treat_player_name_as_achievement() -> None:
    out = parse_left_panel("英雄：51/51 挑战完成")
    assert out.achievement_title is None
    assert out.achievement_unlocked is None


def test_parse_achievement_titles_accepts_unseen_titles_and_multiple_checks() -> None:
    assert parse_achievement_titles("新称号甲 ✓ 新称号乙 ✔") == ("新称号甲", "新称号乙")
