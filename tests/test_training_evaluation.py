from pathlib import Path

from training.scripts.prune_rec_checkpoints import prune_checkpoint_dir


ROOT = Path(__file__).resolve().parents[1]


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
