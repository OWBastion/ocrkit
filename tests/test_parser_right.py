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
