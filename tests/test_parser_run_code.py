from app.parser.run_code import RUN_CODE_MIN_CONFIDENCE, enforce_run_code_confidence, parse_run_code


def test_parse_run_code_normalizes_safe_separator_and_whitespace_variants() -> None:
    parsed = parse_run_code("本局代码：4821 － 7354 — 1926")

    assert parsed.value == "4821-7354-1926"
    assert parsed.status == "ok"
    assert parsed.normalization == ("separator:canonical-hyphen", "whitespace:trimmed")


def test_parse_run_code_requires_the_visible_label_and_complete_numeric_groups() -> None:
    no_label = parse_run_code(
        "版本 26.0613.3 总计耗时 2小时20分38秒 总计阵亡/跳过 114/0 增益/减益/总计 29/49/106 4821-7354-1926"
    )
    malformed = parse_run_code("本局代码：4821-7354-192")
    leading_zero = parse_run_code("Run Code: 0821-7354-1926")
    merged = parse_run_code("本局代码：482173541926")

    assert no_label.status == "missing"
    assert malformed.status == "invalid"
    assert leading_zero.status == "invalid"
    assert merged.status == "invalid"


def test_parse_run_code_marks_conflicting_candidates_ambiguous() -> None:
    parsed = parse_run_code("本局代码：4821-7354-1926 Run Code: 4821-7354-1927")

    assert parsed.value is None
    assert parsed.status == "ambiguous"
    assert parsed.warning == "run_code.ambiguous"


def test_parse_run_code_accepts_a_repeated_identical_candidate() -> None:
    parsed = parse_run_code("本局代码：4821-7354-1926 Run Code: 4821-7354-1926")

    assert parsed.value == "4821-7354-1926"
    assert parsed.status == "ok"


def test_enforce_run_code_confidence_rejects_below_threshold_without_guessing() -> None:
    parsed = enforce_run_code_confidence(parse_run_code("本局代码：4821-7354-1926"), RUN_CODE_MIN_CONFIDENCE - 0.01)

    assert parsed.value is None
    assert parsed.status == "low_confidence"
    assert parsed.warning == "run_code.low_confidence"
