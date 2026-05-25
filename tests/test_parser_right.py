from app.parser.right_panel import parse_right_panel


def test_parse_right_panel() -> None:
    out = parse_right_panel("巴黎：地狱 版本 26.O5O6.1", ["巴黎：地狱", "伊利奥斯"])
    assert out.map_name == "巴黎：地狱"
    assert out.version == "26.0506.1"
