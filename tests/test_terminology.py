from __future__ import annotations

from pathlib import Path

import numpy as np

from app.parser.terminology import (
    DECISION_AMBIGUOUS,
    DECISION_NORMALIZED,
    DECISION_UNCHANGED,
    DECISION_UNRESOLVED,
    Scope,
    TerminologyCatalog,
    load_terminology_rules,
    normalize_roi_text,
    normalize_scope_text,
    scope_for,
)

REAL_RULES = load_terminology_rules(Path("configs/terminology.yaml"))


def test_motivating_example_challenge_stats_normalizes_in_left_panel() -> None:
    result = normalize_roi_text("编益/减益/总计", "1280x720-v6", "left_panel", REAL_RULES)
    assert result.decision == DECISION_NORMALIZED
    assert result.normalized_text == "增益/减益/总计"
    assert result.raw_text == "编益/减益/总计"
    token = result.tokens[0]
    assert token.status == "adopted"
    assert token.match_type == "exact_alias"
    assert token.rule_id == "left_panel.challenge_stats:alias:编益"
    assert result.rules_version == "1"


def test_motivating_example_hero_status_normalizes_in_bottom_left() -> None:
    result = normalize_roi_text("心之钢/移速/减伤/疗%", "1280x720-v6", "bottom_left_hero", REAL_RULES)
    assert result.decision == DECISION_NORMALIZED
    assert result.normalized_text == "心之钢/移速/减伤/治疗%"
    assert result.tokens[-1].rule_id == "bottom_left_hero.hero_status:alias:疗%"


def test_character_confusion_adopted_only_inside_allowed_terms() -> None:
    scope = Scope(
        id="s",
        roi="left_panel",
        layout_versions=frozenset({"1280x720-v6"}),
        allowed_terms=("增益", "减益", "总计"),
        aliases={},
        confusions={"编": "增"},
    )
    adopted = normalize_scope_text("编益", scope)  # type: ignore[arg-type]
    assert adopted.decision == DECISION_NORMALIZED
    assert adopted.normalized_text == "增益"

    off_allowlist = normalize_scope_text("编益X", scope)  # type: ignore[arg-type]
    assert off_allowlist.decision == DECISION_UNRESOLVED
    assert off_allowlist.normalized_text == "编益X"


def test_false_friend_term_is_not_touched() -> None:
    result = normalize_roi_text("减益", "1280x720-v6", "left_panel", REAL_RULES)
    assert result.decision == DECISION_UNCHANGED
    assert result.normalized_text == "减益"


def test_unknown_text_stays_unresolved_or_unchanged() -> None:
    # In-scope token that looks like a confusion but lands off-allowlist.
    unresolved = normalize_roi_text("编编", "1280x720-v6", "left_panel", REAL_RULES)
    assert unresolved.decision == DECISION_UNRESOLVED
    # Ordinary text that is not terminology stays unchanged, never fabricated.
    ordinary = normalize_roi_text("挑战 完成 英雄: 51/51", "1280x720-v6", "left_panel", REAL_RULES)
    assert ordinary.decision == DECISION_UNCHANGED
    assert ordinary.normalized_text == "挑战 完成 英雄: 51/51"


def test_fuzzy_matching_is_constrained_and_ambiguity_detected() -> None:
    scope = Scope(
        id="fuzzy",
        roi="bottom_left_hero",
        layout_versions=frozenset({"1280x720-v6"}),
        allowed_terms=("心之钢", "移速", "减伤", "治疗"),
        aliases={},
        confusions={},
        fuzzy_match=True,
        fuzzy_threshold=0.62,
        fuzzy_min_margin=0.2,
    )
    resolved = normalize_scope_text("心之锅", scope)  # type: ignore[arg-type]
    assert resolved.decision == DECISION_NORMALIZED
    assert resolved.normalized_text == "心之钢"
    assert resolved.tokens[0].match_type == "fuzzy"

    # "益" is equidistant from several two-char terms -> ambiguous, not invented.
    ambiguous_scope = Scope(
        id="amb",
        roi="left_panel",
        layout_versions=frozenset({"1280x720-v6"}),
        allowed_terms=("增益", "减益", "总计"),
        aliases={},
        confusions={},
        fuzzy_match=True,
        fuzzy_threshold=0.5,
        fuzzy_min_margin=0.2,
    )
    ambiguous = normalize_scope_text("益", ambiguous_scope)  # type: ignore[arg-type]
    assert ambiguous.decision == DECISION_AMBIGUOUS
    assert ambiguous.normalized_text == "益"


def test_scope_is_isolated_by_layout_and_roi() -> None:
    # Same text, different ROI: the challenge_stats alias must not fire.
    other_roi = normalize_roi_text("编益", "1280x720-v6", "right_panel", REAL_RULES)
    assert other_roi.decision == DECISION_UNCHANGED
    assert other_roi.normalized_text == "编益"
    # Unknown layout version: pass through, no scope.
    unknown_layout = normalize_roi_text("编益", "9999x999-v9", "left_panel", REAL_RULES)
    assert unknown_layout.scope_id is None
    assert unknown_layout.decision == DECISION_UNCHANGED
    # A second known layout shares the same rule set.
    second_layout = normalize_roi_text("编益", "1280x800-v1", "left_panel", REAL_RULES)
    assert second_layout.decision == DECISION_NORMALIZED
    assert second_layout.normalized_text == "增益"


def test_scope_for_returns_none_outside_declared_layouts() -> None:
    assert scope_for(REAL_RULES, "1280x720-v6", "left_panel") is not None
    assert scope_for(REAL_RULES, "1280x720-v6", "achievement_panel") is None
    assert scope_for(REAL_RULES, "unknown-layout", "left_panel") is None


def test_confusions_must_map_single_characters() -> None:
    from pathlib import Path as _Path
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(
            "rules_version: '1'\n"
            "scopes:\n"
            "  - id: bad\n"
            "    roi: left_panel\n"
            "    layout_versions: ['1280x720-v6']\n"
            "    allowed_terms: ['增益']\n"
            "    confusions:\n"
            "      '编': '增益'\n"
        )
        path = _Path(handle.name)
    try:
        try:
            load_terminology_rules(path)
        except ValueError as exc:
            assert "single characters" in str(exc)
        else:
            raise AssertionError("expected multi-character confusion to be rejected")
    finally:
        path.unlink()


def test_duplicate_scope_ids_rejected() -> None:
    from pathlib import Path as _Path
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(
            "rules_version: '1'\n"
            "scopes:\n"
            "  - id: dup\n"
            "    roi: left_panel\n"
            "    layout_versions: ['1280x720-v6']\n"
            "    allowed_terms: ['增益']\n"
            "  - id: dup\n"
            "    roi: bottom_left_hero\n"
            "    layout_versions: ['1280x720-v6']\n"
            "    allowed_terms: ['心之钢']\n"
        )
        path = _Path(handle.name)
    try:
        try:
            load_terminology_rules(path)
        except ValueError as exc:
            assert "duplicate terminology scope id" in str(exc)
        else:
            raise AssertionError("expected duplicate scope id to be rejected")
    finally:
        path.unlink()


class _TerminologyEngine:
    """Returns the motivating confusion for the left_panel ROI and nothing else."""

    def __init__(self, left_panel_text: str) -> None:
        self.left_panel_text = left_panel_text

    def recognize(self, image: np.ndarray):
        from app.ocr.engine import OcrResult

        if image.shape[:2] == (420, 370):  # left_panel preprocessed at 2x
            return OcrResult(text=self.left_panel_text, confidence=0.93, chunks=[])
        return OcrResult(text="", confidence=0.5, chunks=[])


def _service_context(engine, catalog: TerminologyCatalog | None = REAL_RULES):
    from app.core.context import AppContext
    from app.core.roi_config import load_map_aliases, load_map_names, load_roi_config

    return AppContext(
        roi_config=load_roi_config(Path("configs/roi_1280x720.yaml")),
        map_names=load_map_names(Path("configs/maps.yaml")),
        map_aliases=load_map_aliases(Path("configs/maps.yaml")),
        ocr_engine=engine,
        terminology=catalog,
    )


def test_service_preserves_raw_evidence_and_records_normalization() -> None:
    from app.service import extract_structured

    context = _service_context(_TerminologyEngine("编益/减益/总计"))
    response = extract_structured(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        context.roi_config,
        context.map_names,
        context.map_aliases,
        context.ocr_engine,
        True,
        "request-terminology-1",
        "rapidocr",
        "builtin",
        "1280x720-v6",
        terminology=context.terminology,
    )

    assert response.debug is not None
    # Raw OCR evidence is untouched.
    assert response.debug.raw_text["left_panel"] == "编益/减益/总计"
    normalization = response.debug.terminology_normalization["left_panel"]
    assert normalization.decision == DECISION_NORMALIZED
    assert normalization.normalized_text == "增益/减益/总计"
    assert normalization.scope_id == "left_panel.challenge_stats"
    assert normalization.rules_version == "1"
    assert normalization.tokens[0].rule_id == "left_panel.challenge_stats:alias:编益"
    # Field evidence carries a traceable normalization tag.
    assert "terminology:增益" in response.fields["deaths"].normalization
    # No ambiguous warning for a clean deterministic adoption.
    assert "terminology.left_panel.ambiguous" not in response.warnings


def test_service_warns_when_terminology_is_ambiguous() -> None:
    from app.service import extract_structured

    ambiguous_catalog = TerminologyCatalog(
        rules_version="1",
        scopes=(
            Scope(
                id="left_panel.challenge_stats",
                roi="left_panel",
                layout_versions=frozenset({"1280x720-v6"}),
                allowed_terms=("增益", "减益", "总计"),
                aliases={},
                confusions={},
                fuzzy_match=True,
                fuzzy_threshold=0.5,
                fuzzy_min_margin=0.2,
            ),
        ),
    )
    context = _service_context(_TerminologyEngine("益"), ambiguous_catalog)
    response = extract_structured(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        context.roi_config,
        context.map_names,
        context.map_aliases,
        context.ocr_engine,
        False,
        "request-terminology-2",
        "rapidocr",
        "builtin",
        "1280x720-v6",
        terminology=context.terminology,
    )

    assert response.debug is None
    assert "terminology.left_panel.ambiguous" in response.warnings
    assert response.data is not None
    # The ambiguous token is never invented; parsing falls back to raw text.
    assert response.debug is None or response.debug.raw_text["left_panel"] == "益"
