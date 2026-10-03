from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from training.scripts import evaluate_rec_holdout


def _crops(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = []
    for relative, text in rows:
        crop = tmp_path / relative
        crop.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(crop), np.zeros((4, 4, 3), dtype=np.uint8))
        lines.append(f"{relative}\t{text}\n")
    labels = tmp_path / "holdout.txt"
    labels.write_text("".join(lines), encoding="utf-8")
    return labels


def _engine(monkeypatch: pytest.MonkeyPatch, texts: list[str], seen: list[dict[str, object]]) -> None:
    answers = iter(texts)

    class Wrapper:
        def __init__(self, _config: Path) -> None:
            def run(_image: np.ndarray, **kwargs: object) -> SimpleNamespace:
                seen.append(kwargs)
                text = next(answers)
                return SimpleNamespace(txts=(text,) if text else ())

            self._engine = run

    monkeypatch.setattr(evaluate_rec_holdout, "RapidOcrEngine", Wrapper)


def test_holdout_scores_recognition_only_ignoring_spacing_and_reports_each_region(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = _crops(tmp_path, [
        ("images/holdout/s1/right_panel/000.png", "版本26.1002.3"),
        ("images/holdout/s1/right_panel/001.png", "服务器负载87"),
        ("images/holdout/s1/left_panel/000.png", "1小时53分23秒"),
        ("images/holdout/s1/left_panel/001.png", "总计阵亡/跳过"),
    ])
    seen: list[dict[str, object]] = []
    _engine(monkeypatch, ["版本 26.1002.3", "服务器负载 88", "1 小时53 分23 秒", ""], seen)

    report = evaluate_rec_holdout.evaluate(labels, tmp_path, tmp_path / "rapidocr.yaml")

    assert all(call == {"use_det": False, "use_cls": False} for call in seen)
    assert (report["matched"], report["total"]) == (2, 4)
    assert report["accuracy"] == 0.5
    assert report["by_roi"] == {"left_panel": {"matched": 1, "total": 2}, "right_panel": {"matched": 1, "total": 2}}
    assert "status" not in report  # pass/fail is decided by the paired comparison, not by a fixed number


def test_holdout_evaluation_rejects_crops_outside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = tmp_path / "holdout.txt"
    labels.write_text("../outside.png\t文本\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_rec_holdout, "RapidOcrEngine", lambda _config: SimpleNamespace(_engine=None))

    with pytest.raises(RuntimeError, match="outside the images root"):
        evaluate_rec_holdout.evaluate(labels, tmp_path, tmp_path / "rapidocr.yaml")
