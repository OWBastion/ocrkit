from app.parser.left_panel import parse_left_panel


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
