from app.parser.center_summary import parse_center_summary


def test_parse_center_summary() -> None:
    out = parse_center_summary("祝贺 天树是只臭猫 以 94 次阵亡 & 0 次跳过 耗时 2小时9分59秒 通关")
    assert out.completed is True
    assert out.player == "天树是只臭猫"
    assert out.deaths == 94
    assert out.skips == 0
    assert out.duration_seconds == 7799.0


def test_parse_center_summary_with_separator() -> None:
    out = parse_center_summary("贺训犬大师以91次阵亡&0次跳过·耗时1小时56分51秒通关！")
    assert out.player == "训犬大师"
    assert out.deaths == 91
    assert out.skips == 0
    assert out.duration_seconds == 7011.0


def test_parse_center_summary_extracts_player_without_completion_suffix() -> None:
    out = parse_center_summary("兄贺训犬大师以182次阵亡&0次跳过耗时2小时33分53秒")
    assert out.completed is False
    assert out.player == "训犬大师"


def test_parse_center_summary_without_player_separator() -> None:
    out = parse_center_summary("祝贺训犬大师 92次阵亡&0次跳过耗时1小时26分36秒通关！")
    assert out.player == "训犬大师"
    assert out.deaths == 92
    assert out.skips == 0
    assert out.duration_seconds == 5196.0
