from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import training.studio.app as studio_app
from training.importer.contract import SnapshotMetadata
from training.studio.app import create_app


def _provenance() -> dict[str, object]:
    return {
        "schema_version": 1,
        "importer_version": "1",
        "snapshot": {"snapshot_id": "snap-2026-08-01", "version": "v3", "finalized_at": "2026-08-01T00:00:00Z"},
        "split": {
            "rule_version": "ocrkit-split-v1",
            "split_seed": "ocrkit-v1",
            "holdout_fraction": 0.2,
            "assignment": {"src-1": "train", "src-2": "holdout"},
        },
        "layout_versions": ["1280x720-v6"],
        "code_revision": "abc1234",
        "imported_at": "2026-08-02T00:00:00Z",
        "sources": [
            {"source_id": "src-1", "object_id": "obj-1", "sha256": "0" * 64, "size_bytes": 1, "layout_version": "1280x720-v6"}
        ],
        "annotation_count": 1,
        "labels": {"train": 1, "holdout": 1},
        "warnings": [],
        "label_conflicts": [],
    }


def _import_dir(root: Path, ref: str = "snap-2026-08-01@v3") -> Path:
    import_dir = root / ref
    labels = import_dir / "labels"
    labels.mkdir(parents=True)
    (labels / "train.txt").write_text("images/train/src-1/left_panel.png\t增益\n", encoding="utf-8")
    (labels / "holdout.txt").write_text("images/holdout/src-2/left_panel.png\t减益\n", encoding="utf-8")
    (import_dir / "annotations.jsonl").write_text(
        json.dumps(
            {"annotation_id": "a1", "source_id": "src-1", "split": "train", "roi": "left_panel",
             "ocr_prediction": "编益", "exact_transcription": "增益", "canonical_value": "增益",
             "crop_path": "images/train/src-1/left_panel.png"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (import_dir / "provenance.json").write_text(json.dumps(_provenance(), ensure_ascii=False) + "\n", encoding="utf-8")
    return import_dir


def _client(tmp_path: Path) -> TestClient:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>Studio</main>", encoding="utf-8")
    return TestClient(create_app(tmp_path / "work", frontend, snapshot_import_root=tmp_path / "imports"))


def test_snapshots_lists_materialized_imports_with_provenance(tmp_path: Path) -> None:
    _import_dir(tmp_path / "imports")
    client = _client(tmp_path)

    response = client.get("/api/snapshots")

    assert response.status_code == 200
    assert len(response.json()) == 1
    summary = response.json()[0]
    assert summary["snapshot_id"] == "snap-2026-08-01"
    assert summary["version"] == "v3"
    assert summary["ref"] == "snap-2026-08-01@v3"
    assert summary["train_sources"] == 1
    assert summary["holdout_sources"] == 1
    assert summary["labels"] == {"train": 1, "holdout": 1}
    assert summary["code_revision"] == "abc1234"
    assert summary["warnings"] == []


def test_snapshot_detail_returns_provenance_and_annotations(tmp_path: Path) -> None:
    _import_dir(tmp_path / "imports")
    client = _client(tmp_path)

    response = client.get("/api/snapshots/snap-2026-08-01@v3")

    assert response.status_code == 200
    body = response.json()
    assert body["provenance"]["split"]["assignment"]["src-1"] == "train"
    assert body["summary"]["labels"] == {"train": 1, "holdout": 1}
    annotations = body["annotations"]
    assert len(annotations) == 1
    row = annotations[0]
    assert row["ocr_prediction"] == "编益"
    assert row["exact_transcription"] == "增益"
    assert row["canonical_value"] == "增益"


def test_snapshot_annotations_respects_limit(tmp_path: Path) -> None:
    _import_dir(tmp_path / "imports")
    client = _client(tmp_path)

    response = client.get("/api/snapshots/snap-2026-08-01@v3/annotations", params={"limit": 1})

    assert response.status_code == 200
    assert len(response.json()["annotations"]) == 1
    assert client.get("/api/snapshots/snap-2026-08-01@v3/annotations", params={"limit": 0}).status_code == 422


def test_unknown_snapshot_import_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/snapshots/not-imported@v1").status_code == 404
    assert client.get("/api/snapshots/../escape").status_code == 404


def test_snapshot_import_requires_platform_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCRKIT_PLATFORM_SNAPSHOT_BASE_URL", raising=False)
    monkeypatch.delenv("OCRKIT_PLATFORM_SNAPSHOT_TOKEN", raising=False)
    client = _client(tmp_path)

    response = client.post("/api/snapshots/import", json={"snapshot_id": "snap-2026-08-01"})

    assert response.status_code == 503
    assert "OCRKIT_PLATFORM_SNAPSHOT_BASE_URL" in response.json()["detail"]


def test_snapshot_import_materializes_under_import_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCRKIT_PLATFORM_SNAPSHOT_BASE_URL", "https://platform.example")
    monkeypatch.setenv("OCRKIT_PLATFORM_SNAPSHOT_TOKEN", "token")
    metadata = SnapshotMetadata(
        schema_version=1,
        snapshot_id="snap-2026-08-01",
        version="v3",
        finalized=True,
        finalized_at="2026-08-01T00:00:00Z",
        objects=[
            {"object_id": "obj-1", "kind": "source", "sha256": "0" * 64, "mime_type": "image/png",
             "size_bytes": 1, "source_id": "src-1", "layout_version": "1280x720-v6"}
        ],
    )

    class _FakeClient:
        def fetch_snapshot(self, snapshot_id: str) -> SnapshotMetadata:
            assert snapshot_id == "snap-2026-08-01"
            return metadata

    captured: dict[str, object] = {}

    def fake_import_snapshot(**kwargs):
        captured.update(kwargs)
        output = Path(kwargs["output"])
        labels = output / "labels"
        labels.mkdir(parents=True)
        (labels / "train.txt").write_text("images/train/src-1/left_panel.png\t增益\n", encoding="utf-8")
        (labels / "holdout.txt").write_text("", encoding="utf-8")
        (output / "annotations.jsonl").write_text("", encoding="utf-8")
        provenance = _provenance()
        provenance["labels"] = {"train": 1, "holdout": 0}
        (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False) + "\n", encoding="utf-8")
        return SimpleNamespace(
            snapshot_id="snap-2026-08-01",
            snapshot_version="v3",
            sources=1,
            annotations=1,
            split={"train": 1, "holdout": 1},
            labels={"train": 1, "holdout": 0},
            conflicts=[],
            warnings=[],
            output=str(output),
            workspace=str(Path(kwargs["workspace"])),
        )

    monkeypatch.setattr(studio_app, "HttpSnapshotClient", lambda base_url, token: _FakeClient())
    monkeypatch.setattr(studio_app, "import_platform_snapshot", fake_import_snapshot)
    client = _client(tmp_path)

    response = client.post("/api/snapshots/import", json={"snapshot_id": "snap-2026-08-01", "holdout_fraction": 0.25})

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["snapshot_version"] == "v3"
    assert body["summary"]["snapshot_id"] == "snap-2026-08-01"
    assert captured["output"] == tmp_path / "imports/snap-2026-08-01@v3"
    assert captured["workspace"] == tmp_path / "work/snapshot-workspace/snap-2026-08-01"
    assert captured["holdout_fraction"] == 0.25
    assert captured["resume"] is True


def test_snapshot_import_surfaces_missing_evidence_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCRKIT_PLATFORM_SNAPSHOT_BASE_URL", "https://platform.example")
    monkeypatch.setenv("OCRKIT_PLATFORM_SNAPSHOT_TOKEN", "token")
    metadata = SnapshotMetadata(
        schema_version=1,
        snapshot_id="snap-2026-08-01",
        version="v3",
        finalized=True,
        objects=[
            {"object_id": "obj-1", "kind": "source", "sha256": "0" * 64, "mime_type": "image/png",
             "size_bytes": 1, "source_id": "src-1", "layout_version": "1280x720-v6"}
        ],
    )
    monkeypatch.setattr(studio_app, "HttpSnapshotClient", lambda base_url, token: SimpleNamespace(fetch_snapshot=lambda snapshot_id: metadata))

    def broken_import(**kwargs):
        raise studio_app.MissingSourceError("snapshot evidence unavailable: obj-9")

    monkeypatch.setattr(studio_app, "import_platform_snapshot", broken_import)
    client = _client(tmp_path)

    response = client.post("/api/snapshots/import", json={"snapshot_id": "snap-2026-08-01"})

    assert response.status_code == 422
    assert "obj-9" in response.json()["detail"]


def test_snapshot_smoke_training_uses_import_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _import_dir(tmp_path / "imports")
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123))
    client = _client(tmp_path)

    response = client.post("/api/snapshots/snap-2026-08-01@v3/training/smoke", json={"epochs": 5})

    assert response.status_code == 200
    assert response.json()["status"] == "training"
    assert "--labels-dir" in response.json()["command"]
    assert str(tmp_path / "imports/snap-2026-08-01@v3/labels") in response.json()["command"]

    status = client.get("/api/snapshots/snap-2026-08-01@v3/training")
    assert status.json()["status"] in {"training", "completed_or_failed"}

    latest = json.loads((tmp_path / "work/snapshot-runs/snap-2026-08-01@v3/runs/latest.json").read_text(encoding="utf-8"))
    assert latest["pid"] == 123


def test_snapshot_publication_requires_confirmed_completed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _import_dir(tmp_path / "imports")
    run_root = tmp_path / "work/snapshot-runs/snap-2026-08-01@v3"
    checkpoint = run_root / "runs/smoke-1/checkpoints/best_accuracy"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.with_suffix(".pdparams").write_text("checkpoint", encoding="utf-8")
    (run_root / "runs/latest.json").write_text(
        json.dumps({"status": "completed", "exit_code": 0, "log": str(run_root / "runs/smoke-1/training.log")}),
        encoding="utf-8",
    )
    monkeypatch.setattr(studio_app.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=456))
    client = _client(tmp_path)

    assert client.post("/api/snapshots/snap-2026-08-01@v3/publication", json={"confirmed": False}).status_code == 422
    response = client.post("/api/snapshots/snap-2026-08-01@v3/publication", json={"confirmed": True})

    assert response.status_code == 200
    assert response.json()["status"] == "publishing"
    assert response.json()["command"][-1] == str(checkpoint)


def test_snapshot_lists_resume_checkpoints_from_runs_and_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _import_dir(tmp_path / "imports")
    run_root = tmp_path / "work/snapshot-runs/snap-2026-08-01@v3"
    checkpoint = run_root / "runs/smoke-1/checkpoints/best_accuracy"
    checkpoint.parent.mkdir(parents=True)
    for suffix in (".pdparams", ".pdopt", ".states"):
        (checkpoint.parent / f"best_accuracy{suffix}").write_text("checkpoint", encoding="utf-8")
    client = _client(tmp_path)

    response = client.get("/api/snapshots/snap-2026-08-01@v3/training/checkpoints")

    assert response.status_code == 200
    assert response.json() == [{
        "path": "snapshot:runs/smoke-1/checkpoints/best_accuracy",
        "name": "本快照 · smoke-1/checkpoints/best_accuracy",
    }]


def test_snapshot_resume_checkpoint_rejects_traversal(tmp_path: Path) -> None:
    _import_dir(tmp_path / "imports")
    run_root = studio_app._snapshot_run_root(tmp_path / "work", "snap-2026-08-01@v3")
    with pytest.raises(studio_app.HTTPException):
        studio_app._snapshot_resume_checkpoint(tmp_path / "work", run_root, "snapshot:../outside")
