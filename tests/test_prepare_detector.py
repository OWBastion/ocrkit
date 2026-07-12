from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path("training/scripts/prepare_detector.py")
    spec = importlib.util.spec_from_file_location("prepare_detector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_detector_downloads_and_reuses_verified_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    payload = b"detector"
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"filename": "det.onnx", "url": "https://example.invalid/det.onnx", "sha256": hashlib.sha256(payload).hexdigest()}),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    output = tmp_path / "out" / "det.onnx"
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_kwargs: Path(args[-1]).write_bytes(payload))
    monkeypatch.setattr(module, "__name__", "__main__")
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda _self: type("Args", (), {"lock": lock, "cache_dir": cache, "output": output})())

    module.main()
    output.unlink()
    module.main()
    assert output.read_bytes() == payload


def test_prepare_detector_rejects_bad_download_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"filename": "det.onnx", "url": "https://example.invalid/det.onnx", "sha256": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_kwargs: Path(args[-1]).write_bytes(b"wrong"))
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda _self: type("Args", (), {"lock": lock, "cache_dir": tmp_path / "cache", "output": tmp_path / "out.onnx"})())

    with pytest.raises(SystemExit, match="checksum mismatch"):
        module.main()
