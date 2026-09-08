from pathlib import Path

from training.scripts.prune_rec_checkpoints import discover_checkpoint_dirs, prune_checkpoint_dir


ROOT = Path(__file__).resolve().parents[1]


def test_training_and_release_use_shared_checkpoint_evaluator() -> None:
    evaluator = "training/evaluate_rec_checkpoint.sh"
    assert evaluator in (ROOT / "training/run_rec_smoke.sh").read_text(encoding="utf-8")
    assert evaluator in (ROOT / "training/release_rec_model.sh").read_text(encoding="utf-8")


def test_smoke_training_does_not_keep_every_epoch_checkpoint() -> None:
    text = (ROOT / "training/run_rec_smoke.sh").read_text(encoding="utf-8")
    assert "Global.save_epoch_step=1" not in text
    assert 'Global.save_epoch_step="$((epoch_num + 1))"' in text
    assert "training/scripts/prune_rec_checkpoints.py" in text


def test_prune_rec_checkpoints_keeps_latest_and_best_accuracy(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model").mkdir()
    (checkpoint_dir / "best_model" / "model.pdparams").write_text("dup", encoding="utf-8")
    for name in (
        "best_accuracy.pdparams",
        "best_accuracy.pdopt",
        "best_accuracy.states",
        "latest.pdparams",
        "latest.pdopt",
        "latest.states",
        "iter_epoch_1.pdparams",
        "iter_epoch_1.pdopt",
        "iter_epoch_1.states",
        "config.yml",
        "train.log",
    ):
        (checkpoint_dir / name).write_text(name, encoding="utf-8")

    removed = prune_checkpoint_dir(checkpoint_dir)

    assert {path.name for path in removed} == {
        "best_model",
        "iter_epoch_1.pdparams",
        "iter_epoch_1.pdopt",
        "iter_epoch_1.states",
    }
    remaining = {path.name for path in checkpoint_dir.iterdir()}
    assert remaining == {
        "best_accuracy.pdparams",
        "best_accuracy.pdopt",
        "best_accuracy.states",
        "latest.pdparams",
        "latest.pdopt",
        "latest.states",
        "config.yml",
        "train.log",
    }


def test_discover_checkpoint_dirs_finds_epoch_dumps_and_best_model(tmp_path: Path) -> None:
    first = tmp_path / "runs/smoke-1/checkpoints"
    second = tmp_path / "runs/smoke-2/checkpoints"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "iter_epoch_2.pdparams").write_text("epoch", encoding="utf-8")
    (second / "best_model").mkdir()
    (tmp_path / "unrelated.txt").write_text("skip", encoding="utf-8")

    assert discover_checkpoint_dirs(tmp_path) == [first, second]


def test_checkpoint_evaluator_has_no_r2_upload_commands() -> None:
    text = (ROOT / "training/evaluate_rec_checkpoint.sh").read_text(encoding="utf-8")
    assert "upload_artifacts.py" not in text
    assert "next_model_version.py" not in text
    assert "fixture_report.json" in text
