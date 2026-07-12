from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_release_use_shared_checkpoint_evaluator() -> None:
    evaluator = "training/evaluate_rec_checkpoint.sh"
    assert evaluator in (ROOT / "training/run_rec_smoke.sh").read_text(encoding="utf-8")
    assert evaluator in (ROOT / "training/release_rec_model.sh").read_text(encoding="utf-8")


def test_checkpoint_evaluator_has_no_r2_upload_commands() -> None:
    text = (ROOT / "training/evaluate_rec_checkpoint.sh").read_text(encoding="utf-8")
    assert "upload_artifacts.py" not in text
    assert "next_model_version.py" not in text
    assert "fixture_report.json" in text
