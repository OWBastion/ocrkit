from app.parser.normalize import normalize_numeric_text, parse_time_to_seconds


def test_parse_time_noisy_hms() -> None:
    assert parse_time_to_seconds("2小时9分S9秒") == 7799.0


def test_parse_time_mmss() -> None:
    assert parse_time_to_seconds("12:30.5") == 750.5


def test_numeric_normalize() -> None:
    assert normalize_numeric_text("26.O5O6.l") == "26.0506.1"
