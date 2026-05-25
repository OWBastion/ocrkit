from app.parser.center_summary import parse_center_summary


def test_parse_center_summary() -> None:
    out = parse_center_summary("祝贺 天树是只臭猫 以 94 次阵亡 & 0 次跳过 耗时 2小时9分59秒 通关")
    assert out.completed is True
    assert out.player == "天树是只臭猫"
    assert out.deaths == 94
    assert out.skips == 0
    assert out.duration_seconds == 7799.0
