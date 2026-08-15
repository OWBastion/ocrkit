from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import training.importer.importer as importer_module
from training.importer.client import (
    HttpSnapshotClient,
    ObjectUnavailableError,
    SnapshotAuthError,
    SnapshotContractError,
    SnapshotNotFoundError,
)
from training.importer.contract import (
    AnnotationsPayload,
    ReviewedAnnotation,
    SnapshotMetadata,
    SnapshotObject,
)
from training.importer.importer import (
    MissingSourceError,
    default_layout_configs,
    import_snapshot,
)
from training.importer.split import source_split, split_sources

LAYOUT = "1280x720-v6"


def _png_bytes(width: int = 320, height: int = 180, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _source_object(source_id: str, object_id: str, data: bytes) -> SnapshotObject:
    return SnapshotObject(
        object_id=object_id,
        kind="source",
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type="image/png",
        size_bytes=len(data),
        source_id=source_id,
        layout_version=LAYOUT,
    )


def _crop_object(annotation_id: str, object_id: str, data: bytes) -> SnapshotObject:
    return SnapshotObject(
        object_id=object_id,
        kind="crop",
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type="image/png",
        size_bytes=len(data),
        annotation_id=annotation_id,
    )


class FakeSnapshotClient:
    def __init__(self, metadata: SnapshotMetadata, payload: AnnotationsPayload, objects: dict[str, bytes], fail_objects: set[str] | None = None) -> None:
        self.metadata = metadata
        self.payload = payload
        self.objects = objects
        self.fail_objects = fail_objects or set()
        self.downloads: list[str] = []

    def fetch_snapshot(self, snapshot_id: str) -> SnapshotMetadata:
        assert snapshot_id == self.metadata.snapshot_id
        return self.metadata

    def fetch_annotations(self, snapshot_id: str) -> AnnotationsPayload:
        assert snapshot_id == self.metadata.snapshot_id
        return self.payload

    def download_object(self, object_id: str) -> bytes:
        if object_id in self.fail_objects:
            raise ObjectUnavailableError(f"object not available: {object_id}")
        if object_id not in self.objects:
            raise ObjectUnavailableError(f"object not available: {object_id}")
        self.downloads.append(object_id)
        return self.objects[object_id]


def _annotation(annotation_id: str, source_id: str, roi: str = "left_panel", transcription: str = "增益", **overrides) -> ReviewedAnnotation:
    values = {
        "annotation_id": annotation_id,
        "source_id": source_id,
        "layout_version": LAYOUT,
        "roi": roi,
        "field": "challenge_stats",
        "ocr_prediction": "编益",
        "exact_transcription": transcription,
        "canonical_value": "增益",
    }
    values.update(overrides)
    return ReviewedAnnotation(**values)


def _snapshot(sources: list[SnapshotObject], snapshot_id: str = "snap-2026-08-01") -> SnapshotMetadata:
    return SnapshotMetadata(
        schema_version=1,
        snapshot_id=snapshot_id,
        version="v3",
        finalized=True,
        finalized_at="2026-08-01T00:00:00Z",
        objects=sources,
    )


def _payload(annotations: list[ReviewedAnnotation], snapshot_id: str = "snap-2026-08-01") -> AnnotationsPayload:
    return AnnotationsPayload(schema_version=1, snapshot_id=snapshot_id, annotations=annotations)


def _fake_rust_crop_sources(sources_by_layout, workspace: Path, layout_config_paths):
    """Stand-in for the Rust crop-batch that writes the same output layout."""
    for layout_version, sources in sources_by_layout.items():
        group_out = workspace / f".rust-crops-{layout_version}"
        manifest_sources = []
        for source, split in sources:
            roi_config = default_layout_configs()[layout_version][0]
            for roi_name in roi_config.rois:
                destination = group_out / "images" / split / str(source.source_id) / f"{roi_name}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(_png_bytes(seed=hash(source.source_id) % 1000))
            manifest_sources.append({"source_id": source.source_id, "split": split, "rois": {}})
        (group_out / "crop_manifest.json").write_text(
            json.dumps({"schema_version": "1", "layout_version": layout_version, "sources": manifest_sources}),
            encoding="utf-8",
        )
    merged = workspace / "merged-crop-manifest.json"
    merged.write_text(json.dumps({"schema_version": "1", "layout_versions": sorted(sources_by_layout), "sources": []}), encoding="utf-8")
    return merged


@pytest.fixture
def layout_configs():
    return default_layout_configs()


class TestContract:
    def test_snapshot_metadata_rejects_extra_platform_metadata(self) -> None:
        data = {
            "schema_version": 1,
            "snapshot_id": "s1",
            "version": "v1",
            "finalized": True,
            "objects": [{"object_id": "o1", "kind": "source", "sha256": "0" * 64, "mime_type": "image/png", "size_bytes": 1, "source_id": "src-1", "layout_version": LAYOUT}],
        }
        data["qq_id"] = "123456789"
        with pytest.raises(Exception, match="Extra inputs are not permitted"):
            SnapshotMetadata.model_validate(data)

    def test_annotation_rejects_conflicting_crop_sources(self) -> None:
        with pytest.raises(Exception, match="both box and crop_object_id"):
            ReviewedAnnotation(
                annotation_id="a1",
                source_id="src-1",
                layout_version=LAYOUT,
                roi="left_panel",
                exact_transcription="增益",
                box=[[0, 0], [10, 0], [10, 4], [0, 4]],
                crop_object_id="obj-crop",
            )

    def test_annotation_requires_a_crop_source(self) -> None:
        with pytest.raises(Exception, match="needs a crop source"):
            ReviewedAnnotation(
                annotation_id="a1",
                source_id="src-1",
                layout_version=LAYOUT,
                exact_transcription="增益",
            )

    def test_annotations_must_belong_to_the_same_snapshot(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        source = _source_object("src-1", "obj-1", _png_bytes())
        client = FakeSnapshotClient(
            _snapshot([source]),
            _payload([_annotation("a1", "src-1")], snapshot_id="different-snapshot"),
            {"obj-1": _png_bytes()},
        )
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        with pytest.raises(SnapshotContractError, match="belongs to"):
            import_snapshot(
                client=client,
                snapshot_id="snap-2026-08-01",
                workspace=tmp_path / "workspace",
                output=tmp_path / "output",
                layout_configs=layout_configs,
                code_revision="test",
            )


class TestImport:
    def _client(self, objects: dict[str, bytes] | None = None, fail: set[str] | None = None) -> FakeSnapshotClient:
        source_data = {f"src-{index}": _png_bytes(seed=index) for index in range(1, 4)}
        objects = objects or {}
        sources = []
        for index, (source_id, data) in enumerate(source_data.items(), start=1):
            objects[f"obj-{index}"] = data
            sources.append(_source_object(source_id, f"obj-{index}", data))
        annotations = [
            _annotation("a1", "src-1", transcription="增益"),
            _annotation("a2", "src-2", transcription="减益"),
        ]
        return FakeSnapshotClient(_snapshot(sources), _payload(annotations), objects, fail)

    def test_import_materializes_sources_annotations_labels_and_provenance(
        self, tmp_path: Path, layout_configs, monkeypatch
    ) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        client = self._client()
        report = import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test-rev",
        )

        assert report.sources == 3
        assert report.annotations == 2
        assert report.split["train"] + report.split["holdout"] == 3
        assert report.labels["train"] + report.labels["holdout"] == 2
        assert report.conflicts == []
        # Every snapshot member was downloaded exactly once.
        assert len(client.downloads) == 3
        assert sorted(client.downloads) == ["obj-1", "obj-2", "obj-3"]

        output = tmp_path / "output"
        assert (output / "labels/train.txt").is_file() or (output / "labels/holdout.txt").is_file()
        annotations = [json.loads(line) for line in (output / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(annotations) == 2
        row = annotations[0]
        # OCR prediction, reviewed transcription, and canonical value stay distinct.
        assert row["ocr_prediction"] == "编益"
        assert row["exact_transcription"] == "增益"
        assert row["canonical_value"] == "增益"
        assert row["crop_path"].startswith(("images/", "crops/"))

        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["snapshot"] == {"snapshot_id": "snap-2026-08-01", "version": "v3", "finalized_at": "2026-08-01T00:00:00Z"}
        assert provenance["code_revision"] == "test-rev"
        assert provenance["split"]["rule_version"] == "ocrkit-split-v1"
        assert set(provenance["split"]["assignment"]) == {"src-1", "src-2", "src-3"}
        assert provenance["split"]["assignment"]["src-1"] in {"train", "holdout"}
        assert provenance["sources"][0]["sha256"]

    def test_import_is_immutable_and_refuses_overwrite(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        client = self._client()
        import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        with pytest.raises(FileExistsError):
            import_snapshot(
                client=client,
                snapshot_id="snap-2026-08-01",
                workspace=tmp_path / "workspace-2",
                output=tmp_path / "output",
                layout_configs=layout_configs,
                code_revision="test",
            )

    def test_resume_reuses_verified_downloads(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        client = self._client()
        workspace = tmp_path / "workspace"
        import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=workspace,
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        # The workspace survives; a fresh materialization reuses verified objects.
        import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=workspace,
            output=tmp_path / "output-2",
            layout_configs=layout_configs,
            code_revision="test",
            resume=True,
        )
        assert len(client.downloads) == 3  # no second download round

    def test_missing_evidence_fails_explicitly_without_output(
        self, tmp_path: Path, layout_configs, monkeypatch
    ) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        client = self._client(fail={"obj-2"})
        with pytest.raises(MissingSourceError, match="obj-2"):
            import_snapshot(
                client=client,
                snapshot_id="snap-2026-08-01",
                workspace=tmp_path / "workspace",
                output=tmp_path / "output",
                layout_configs=layout_configs,
                code_revision="test",
            )
        assert not (tmp_path / "output").exists()

    def test_checksum_mismatch_is_not_silently_accepted(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        source = _source_object("src-1", "obj-1", _png_bytes())
        client = FakeSnapshotClient(
            _snapshot([source]),
            _payload([_annotation("a1", "src-1")]),
            {"obj-1": b"tampered bytes"},
        )
        with pytest.raises(Exception, match="checksum mismatch"):
            import_snapshot(
                client=client,
                snapshot_id="snap-2026-08-01",
                workspace=tmp_path / "workspace",
                output=tmp_path / "output",
                layout_configs=layout_configs,
                code_revision="test",
            )
        assert not (tmp_path / "output").exists()

    def test_non_finalized_snapshot_is_rejected_before_download(self, tmp_path: Path, layout_configs) -> None:
        from training.importer.client import SnapshotNotFinalizedError

        source = _source_object("src-1", "obj-1", _png_bytes())
        metadata = _snapshot([source])
        metadata = SnapshotMetadata(**{**metadata.model_dump(), "finalized": False})
        client = FakeSnapshotClient(metadata, _payload([_annotation("a1", "src-1")]), {"obj-1": _png_bytes()})
        with pytest.raises(SnapshotNotFinalizedError):
            import_snapshot(
                client=client,
                snapshot_id="snap-2026-08-01",
                workspace=tmp_path / "workspace",
                output=tmp_path / "output",
                layout_configs=layout_configs,
                code_revision="test",
            )
        assert client.downloads == []

    def test_source_level_split_prevents_same_screenshot_leakage(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        client = self._client()
        import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        provenance = json.loads((tmp_path / "output/provenance.json").read_text(encoding="utf-8"))
        assignment = provenance["split"]["assignment"]
        annotations = [json.loads(line) for line in (tmp_path / "output/annotations.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in annotations:
            # Every crop of a source lands in the same split as its source.
            expected = "holdout" if assignment[row["source_id"]] == "holdout" else "train"
            assert row["split"] == expected
            assert row["crop_path"].split("/")[1] == expected
        # Deterministic across runs.
        assert source_split("src-1", "snap-2026-08-01", 0.2) == source_split("src-1", "snap-2026-08-01", 0.2)
        assert split_sources(["src-1"], "snap-2026-08-01", 0.2) == split_sources(["src-1"], "snap-2026-08-01", 0.2)

    def test_label_conflicts_are_reported_and_excluded(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        monkeypatch.setattr(importer_module, "_rust_crop_sources", _fake_rust_crop_sources)
        source = _source_object("src-1", "obj-1", _png_bytes())
        annotations = [
            _annotation("a1", "src-1", transcription="增益"),
            _annotation("a2", "src-1", transcription="减益"),
        ]
        client = FakeSnapshotClient(_snapshot([source]), _payload(annotations), {"obj-1": _png_bytes()})
        report = import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        assert len(report.conflicts) == 1
        assert report.conflicts[0]["annotation_ids"] == ["a1", "a2"]
        assert report.labels["train"] + report.labels["holdout"] == 0
        assert report.warnings and "label conflicts" in report.warnings[0]
        # The reviewed annotations themselves are still materialized as evidence.
        rows = [json.loads(line) for line in (tmp_path / "output/annotations.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2

    def test_line_crop_annotations_do_not_require_rust(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        called = {"rust": False}

        def exploding_rust(*args, **kwargs):
            called["rust"] = True
            raise AssertionError("rust should not run for box annotations")

        monkeypatch.setattr(importer_module, "_rust_crop_sources", exploding_rust)
        source = _source_object("src-1", "obj-1", _png_bytes(width=1280, height=720, seed=7))
        annotation = _annotation(
            "a1",
            "src-1",
            roi="left_panel",
            transcription="增益",
            box=[[40, 20], [120, 20], [120, 30], [40, 30]],
        )
        client = FakeSnapshotClient(_snapshot([source]), _payload([annotation]), {"obj-1": _png_bytes(width=1280, height=720, seed=7)})
        report = import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        assert called["rust"] is False
        assert report.labels["train"] + report.labels["holdout"] == 1
        crops = list((tmp_path / "output/crops").rglob("*.png"))
        assert len(crops) == 1
        crop = cv2.imread(str(crops[0]))
        assert crop is not None and crop.shape[0] == 10  # 20..30 y-span

    def test_pre_crop_annotations_use_platform_samples(self, tmp_path: Path, layout_configs, monkeypatch) -> None:
        called = {"rust": False}
        monkeypatch.setattr(importer_module, "_rust_crop_sources", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no rust")))
        pre_crop = _png_bytes(seed=11)
        source = _source_object("src-1", "obj-1", _png_bytes(seed=1))
        crop_object = _crop_object("a1", "obj-crop", pre_crop)
        annotation = _annotation(
            "a1",
            "src-1",
            transcription="增益",
            crop_object_id="obj-crop",
        )
        client = FakeSnapshotClient(
            _snapshot([source, crop_object]),
            _payload([annotation]),
            {"obj-1": _png_bytes(seed=1), "obj-crop": pre_crop},
        )
        report = import_snapshot(
            client=client,
            snapshot_id="snap-2026-08-01",
            workspace=tmp_path / "workspace",
            output=tmp_path / "output",
            layout_configs=layout_configs,
            code_revision="test",
        )
        assert called["rust"] is False
        assert report.labels["train"] + report.labels["holdout"] == 1
        samples = list((tmp_path / "output/crops").rglob("*.png"))
        assert len(samples) == 1
        assert hashlib.sha256(samples[0].read_bytes()).hexdigest() == hashlib.sha256(pre_crop).hexdigest()


class TestHttpClient:
    class _StubbedClient(HttpSnapshotClient):
        def __init__(self, error: Exception | None = None, body: bytes = b"{}") -> None:
            super().__init__("https://example.test", "token")
            self.error = error
            self.body = body

        def _urlopen(self, req):
            if self.error is not None:
                raise self.error
            body = self.body
            return type("Response", (), {"__enter__": lambda self: self, "__exit__": lambda *a: None, "read": lambda self: body})()

    def test_auth_error_is_mapped(self) -> None:
        from urllib import error as url_error

        client = self._StubbedClient(error=url_error.HTTPError("url", 401, "unauthorized", {}, None))
        with pytest.raises(SnapshotAuthError):
            client.fetch_snapshot("s1")

    def test_not_found_is_mapped(self) -> None:
        from urllib import error as url_error

        client = self._StubbedClient(error=url_error.HTTPError("url", 404, "missing", {}, None))
        with pytest.raises(SnapshotNotFoundError):
            client.fetch_snapshot("missing-snapshot")

    def test_object_download_404_becomes_unavailable(self) -> None:
        from urllib import error as url_error

        client = self._StubbedClient(error=url_error.HTTPError("url", 404, "missing", {}, None))
        with pytest.raises(ObjectUnavailableError):
            client.download_object("obj-1")

    def test_invalid_metadata_json_is_contract_error(self) -> None:
        client = self._StubbedClient(body=b"not json")
        with pytest.raises(SnapshotContractError, match="not valid JSON"):
            client.fetch_snapshot("s1")
