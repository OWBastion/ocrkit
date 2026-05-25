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
