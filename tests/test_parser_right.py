from app.parser.right_panel import parse_right_panel


def test_parse_right_panel_fixed_format_and_version() -> None:
    out = parse_right_panel("巴黎：地狱 版本 26.O5O6.1", ["巴黎", "伊利奥斯"])
    assert out.map_name == "巴黎"
    assert out.difficulty == "地狱"
    assert out.version == "26.0506.1"


def test_parse_right_panel_colon_variant() -> None:
    out = parse_right_panel("国王大道:传奇 版本 26.0513.6", ["国王大道", "伊利奥斯"])
    assert out.map_name == "国王大道"
    assert out.difficulty == "传奇"


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


def test_parse_right_panel_stock_version_variant() -> None:
    out = parse_right_panel("伊利奥斯：地狱 股本26.0513.6", ["巴黎", "伊利奥斯"])
    assert out.version == "26.0513.6"


def test_parse_right_panel_clothes_version_variant() -> None:
    out = parse_right_panel("哈瓦那：地狱 服本26.0518.1", ["哈瓦那"])
    assert out.version == "26.0518.1"


def test_parse_right_panel_restores_version_separators() -> None:
    out = parse_right_panel("尼泊尔：地狱 版本26.06102", ["尼泊尔"])
    assert out.version == "26.0610.2"


def test_parse_right_panel_runasapi() -> None:
    out = parse_right_panel("鲁纳塞彼：地狱 版本26.0621.3", ["努巴尼", "鲁纳塞彼"])
    assert out.map_name == "鲁纳塞彼"
