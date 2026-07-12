from __future__ import annotations

from pathlib import Path

from training.scripts.prepare_rapidocr_config import main


def test_prepare_rapidocr_config_points_to_artifact(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    output = tmp_path / "release.yaml"
    artifact = tmp_path / "artifact"
    template.write_text("Det:\n  model_path: null\nRec:\n  model_path: null\n  rec_keys_path: null\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_rapidocr_config.py", "--template", str(template), "--artifact-dir", str(artifact), "--output", str(output)],
    )

    main()

    text = output.read_text(encoding="utf-8")
    assert str(artifact.resolve() / "det.onnx") in text
    assert str(artifact.resolve() / "rec.onnx") in text
    assert str(artifact.resolve() / "rec_dict.txt") in text
