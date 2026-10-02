from app.parser.run_code import RUN_CODE_MIN_CONFIDENCE, enforce_run_code_confidence, looks_like_run_code_value, parse_run_code


def test_parse_run_code_normalizes_safe_separator_and_whitespace_variants() -> None:
    parsed = parse_run_code("本局代码：4821 － 7354 — 1926")

    assert parsed.value == "4821-7354-1926"
    assert parsed.status == "ok"
    assert parsed.normalization == ("separator:canonical-hyphen", "whitespace:trimmed")


def test_parse_run_code_requires_complete_numeric_groups_with_or_without_the_label() -> None:
    malformed = parse_run_code("本局代码：4821-7354-192")
    leading_zero = parse_run_code("Run Code: 0821-7354-1926")
    merged = parse_run_code("本局代码：482173541926")

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


def test_looks_like_run_code_value_accepts_a_crop_without_its_label() -> None:
    assert looks_like_run_code_value("9811-8811-2854") is True
    assert looks_like_run_code_value("9811 - 8811 — 2854") is True
    assert looks_like_run_code_value("保持距离(31秒)") is False


def test_enforce_run_code_confidence_rejects_below_threshold_without_guessing() -> None:
    parsed = enforce_run_code_confidence(parse_run_code("本局代码：4821-7354-1926"), RUN_CODE_MIN_CONFIDENCE - 0.01)

    assert parsed.value is None
    assert parsed.status == "low_confidence"
    assert parsed.warning == "run_code.low_confidence"


def test_parse_run_code_accepts_the_unlabeled_code_of_the_current_hud() -> None:
    parsed = parse_run_code("8385-4192-4002")

    assert parsed.value == "8385-4192-4002"
    assert parsed.status == "ok"
    assert parsed.normalization == ("label:absent",)


def test_unlabeled_run_code_stays_strict() -> None:
    assert parse_run_code("版本 26.0613.3 29/49/106 114/0").status == "missing"
    assert parse_run_code("0821-7354-1926").status == "missing"
    assert parse_run_code("8385-4192-400").status == "missing"
    assert parse_run_code("8385419240021").status == "missing"
    conflicting = parse_run_code("8385-4192-4002 8385-4192-4003")
    assert (conflicting.value, conflicting.status, conflicting.warning) == (None, "ambiguous", "run_code.ambiguous")
