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


def test_parse_left_panel_does_not_use_boost_stats_for_deaths_skips() -> None:
    out = parse_left_panel("总计阵亡跳过 9210 增益/减益/总计 30/29169")
    assert out.total_deaths is None
    assert out.total_skips is None


def test_parse_achievement_titles_with_checkmark() -> None:
    assert parse_achievement_titles("钢门 ✓") == ("钢门",)


def test_parse_left_panel_does_not_treat_player_name_as_achievement() -> None:
    out = parse_left_panel("英雄：51/51 挑战完成")
    assert out.achievement_title is None
    assert out.achievement_unlocked is None


def test_clear_time_is_read_when_the_banner_hides_its_label() -> None:
    # From recent screenshots: the "祝贺" banner overlaps the stats block and cuts "通关总计耗时" short.
    text = "下一个英雄 挑战完成 总计阵亡/跳过 98/0 增益/减益/总计 31/29/85 心之钢/移速/减伤/治疗% 贺ER 1小时53分10秒 1001-6090-7475"
    parsed = parse_left_panel(text)

    assert (parsed.clear_time, parsed.clear_time_seconds) == ("1小时53分10秒", 6790.0)
    assert (parsed.total_deaths, parsed.total_skips) == (98, 0)


def test_a_labeled_clear_time_still_wins_and_partial_values_are_not_invented() -> None:
    assert parse_left_panel("通关总计耗时 40分19秒 8832-4542-5798").clear_time_seconds == 2419.0
    assert parse_left_panel("通关总计耗时 总计阵亡/跳过 12/0").clear_time_seconds is None
    assert parse_left_panel("祝贺 3小时 7秒 1234-5678-9012").clear_time_seconds is None


def test_left_panel_reports_the_achievement_disabled_notice() -> None:
    text = "下一个英雄 挑战完成 增益0·减益0·总计1 之钢0·移速0·减伤0·治疗0% 成就挑战已禁用"
    assert parse_left_panel(text).achievement_challenge_disabled is True


def test_left_panel_without_the_notice_leaves_it_unknown() -> None:
    assert parse_left_panel("英雄：51/51 挑战完成").achievement_challenge_disabled is None
