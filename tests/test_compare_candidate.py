from __future__ import annotations

from typing import Any

from training.scripts.compare_candidate import decide


def _measured(fields: dict[str, tuple[int, int]], run_code: float = 1.0, holdout: float | None = 0.9) -> dict[str, Any]:
    matched = sum(m for m, _ in fields.values())
    total = sum(t for _, t in fields.values())
    return {
        "fixture": {
            "field_accuracy": matched / total,
            "field_metrics": {name: {"matched": m, "total": t} for name, (m, t) in fields.items()},
            "run_code": {"field_accuracy": run_code},
        },
        "holdout": None if holdout is None else {"accuracy": holdout},
    }


STABLE = _measured({"map_name": (38, 40), "duration_seconds": (36, 40)}, holdout=0.94)


def test_equal_or_better_than_the_model_in_service_passes() -> None:
    assert decide(_measured({"map_name": (39, 40), "duration_seconds": (36, 40)}, holdout=0.94), STABLE) == ("passed", [])


def test_a_single_regressed_field_fails_even_when_the_total_improves() -> None:
    verdict, reasons = decide(_measured({"map_name": (40, 40), "duration_seconds": (35, 40)}, holdout=0.95), STABLE)
    assert verdict == "failed"
    assert reasons == ["regresses duration_seconds: 35/40 vs 36/40 on fixtures"]


def test_worse_holdout_recognition_fails() -> None:
    verdict, reasons = decide(_measured({"map_name": (38, 40), "duration_seconds": (36, 40)}, holdout=0.92), STABLE)
    assert verdict == "failed" and "holdout recognition 0.9200 is below" in reasons[0]


def test_run_code_fixtures_must_stay_exact_with_or_without_a_baseline() -> None:
    candidate = _measured({"map_name": (40, 40), "duration_seconds": (40, 40)}, run_code=0.9)
    assert decide(candidate, STABLE)[0] == "failed"
    assert decide(candidate, None)[0] == "failed"


def test_without_a_baseline_only_the_exact_contract_applies() -> None:
    assert decide(_measured({"map_name": (1, 40)}), None) == ("no_baseline", [])


def test_missing_holdout_on_either_side_is_not_compared() -> None:
    assert decide(_measured({"map_name": (38, 40), "duration_seconds": (36, 40)}, holdout=None), STABLE) == ("passed", [])
