from ocrkit.models import Extracted, OcrTexts
from ocrkit.rules import TitleRules, evaluate_titles


def _texts() -> OcrTexts:
    return OcrTexts(top_bar="", center_banner="", left_panel="")


def test_award_flawless_and_hell_titles() -> None:
    extracted = Extracted(
        passed=True,
        player_name="A",
        time_sec=6000,
        deaths=10,
        skips=0,
        map_label="伊利奥斯",
        difficulty="地狱",
        ocr_texts=_texts(),
    )
    rules = TitleRules(
        version="v",
        loaded_at_epoch=0,
        active_title_keys={"FLAWLESS", "TRAVELER_HELL", "DOMINATOR", "CONQUEROR"},
        map_labels=["伊利奥斯"],
    )
    result = evaluate_titles(extracted, rules)
    assert "FLAWLESS" in result.awarded_keys
    assert "TRAVELER_HELL" in result.awarded_keys
    assert "DOMINATOR" in result.awarded_keys
    assert "CONQUEROR" in result.not_awarded_keys


def test_not_evaluated_when_missing_fields() -> None:
    extracted = Extracted(
        passed=True,
        player_name=None,
        time_sec=None,
        deaths=None,
        skips=None,
        map_label=None,
        difficulty=None,
        ocr_texts=_texts(),
    )
    rules = TitleRules(
        version="v",
        loaded_at_epoch=0,
        active_title_keys={"FLAWLESS", "DODGE_ULTIMATE", "SPEEDRUN", "CONQUEROR"},
        map_labels=[],
    )
    result = evaluate_titles(extracted, rules)
    assert set(result.not_evaluated_keys) == {"CONQUEROR", "DODGE_ULTIMATE", "FLAWLESS", "SPEEDRUN"}
