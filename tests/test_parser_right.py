from app.parser.right_panel import parse_real_player, parse_right_panel


def test_parse_right_panel_fixed_format_and_version() -> None:
    out = parse_right_panel("巴黎：地狱 版本 26.O5O6.1", ["巴黎", "伊利奥斯"])
    assert out.map_name == "巴黎"
    assert out.difficulty == "地狱"
    assert out.version == "26.0506.1"
    assert out.map_variant is None


def test_parse_right_panel_classic_map_variant() -> None:
    out = parse_right_panel("巴黎·经典版：地狱 版本 26.0703.1", ["巴黎", "伊利奥斯"])

    assert out.map_name == "巴黎"
    assert out.difficulty == "地狱"
    assert out.version == "26.0703.1"
    assert out.map_variant == "classic"


def test_parse_right_panel_map_ocr_noise() -> None:
    out = parse_right_panel("国王大 道：困难", ["国王大道", "暴雪世界"])
    assert out.map_name == "国王大道"
    assert out.difficulty == "困难"


def test_parse_right_panel_fps_noise_regression() -> None:
    text = "FPS:160. 退：21MS IND:23MS 4:30PM 巴黎：地狱 [随机事件5.0】 距自动重开1小时49分57秒 版本26.0506.1"
    out = parse_right_panel(text, ["巴黎", "伊利奥斯"])
    assert out.map_name == "巴黎"
    assert out.difficulty == "地狱"
    assert out.version == "26.0506.1"


def test_parse_right_panel_map_alias() -> None:
    out = parse_right_panel("沃斯卡娅：地狱", ["沃斯卡娅工业区"], {"沃斯卡娅": "沃斯卡娅工业区"})
    assert out.map_name == "沃斯卡娅工业区"


def test_parse_right_panel_restores_mode_name_with_lost_leading_character() -> None:
    assert parse_right_panel("好莱坞：地狱 [机事件5.0]", ["好莱坞"]).mode == "随机事件5.0"
    assert parse_right_panel("好莱坞：地狱 [随机事件5.0]", ["好莱坞"]).mode == "随机事件5.0"
    assert parse_right_panel("好莱坞：地狱 [未知模式1.0]", ["好莱坞"]).mode == "未知模式1.0"


def test_real_player_is_the_line_above_the_server_load_label() -> None:
    assert parse_real_player("云雀 服务器负载47") == "云雀"
    assert parse_real_player("  Lark   Wing \n服务器负载 47") == "Lark Wing"


def test_real_player_ignores_a_run_code_read_on_the_same_line() -> None:
    assert parse_real_player("9700-9278-1503 云雀 服务器负载84") == "云雀"
    assert parse_real_player("3217-9535-5179 服务器负载101") is None


def test_real_player_is_absent_when_the_hud_prints_no_name_or_no_label() -> None:
    assert parse_real_player("服务器负载62") is None
    assert parse_real_player("云雀") is None
    assert parse_real_player("") is None
