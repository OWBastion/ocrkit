from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.storage.r2_client import ObjectAccessDeniedError
from training.studio import r2
from training.studio.r2 import StudioR2Store


def configured_settings(**overrides):
    return Settings(
        _env_file=None,
        **{
            "r2_endpoint_url": "https://example.invalid",
            "r2_access_key_id": "test-access",
            "r2_secret_access_key": "test-secret",
            "r2_default_bucket": "models",
            "r2_allowed_buckets": "models,evidence",
            "model_manifest_key": "model/manifest.json",
            "studio_r2_bucket": "",
            "studio_r2_allowed_prefixes": "",
            **overrides,
        },
    )


def test_reuses_existing_credentials_and_unique_evidence_bucket(monkeypatch):
    monkeypatch.setattr(r2, "settings", configured_settings())
    store = StudioR2Store.from_settings()
    assert store is not None
    assert store.bucket == "evidence"
    assert store.allowed_prefixes == ("uploads/submissions/",)
    assert store.object_store.access_key_id == "test-access"
    assert store.object_store.secret_access_key == "test-secret"
    with pytest.raises(ObjectAccessDeniedError):
        store.list_images("models/")


def test_explicit_studio_storage_overrides_shared_settings(monkeypatch):
    monkeypatch.setattr(r2, "settings", configured_settings(studio_r2_bucket="custom-evidence", studio_r2_allowed_prefixes="legacy/images/, uploads/submissions/"))
    store = StudioR2Store.from_settings()
    assert store is not None
    assert store.bucket == "custom-evidence"
    assert store.allowed_prefixes == ("legacy/images/", "uploads/submissions/")


@pytest.mark.parametrize("allowed", ["models", "models,evidence,other"])
def test_does_not_guess_model_or_ambiguous_evidence_bucket(monkeypatch, allowed):
    monkeypatch.setattr(r2, "settings", configured_settings(r2_allowed_buckets=allowed))
    assert StudioR2Store.from_settings() is None


def test_reuses_default_bucket_when_not_reserved_for_models(monkeypatch):
    monkeypatch.setattr(r2, "settings", configured_settings(r2_default_bucket="evidence", model_manifest_key=""))
    store = StudioR2Store.from_settings()
    assert store is not None
    assert store.bucket == "evidence"


def test_recent_filter_keeps_cursor_through_empty_lexicographic_page():
    pages = {
        None: {"Contents": [{"Key": "uploads/submissions/a/old.png", "Size": 100, "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc)}], "IsTruncated": True, "NextContinuationToken": "page-2"},
        "page-2": {"Contents": [{"Key": "uploads/submissions/z/new.png", "Size": 100, "LastModified": datetime(2026, 10, 1, tzinfo=timezone.utc)}, {"Key": "uploads/submissions/z/unknown.png", "Size": 100}], "IsTruncated": False},
    }
    store = StudioR2Store(SimpleNamespace(list_objects=lambda bucket, prefix, cursor, max_keys: pages[cursor]), "evidence", ("uploads/submissions/",), 200, 1000)
    first = store.list_images("uploads/submissions/", since="2026-09-03")
    assert first["objects"] == []
    assert first["next_cursor"] == "page-2"
    second = store.list_images("uploads/submissions/", first["next_cursor"], since="2026-09-03")
    assert [item["key"] for item in second["objects"]] == ["uploads/submissions/z/new.png"]
    assert second["next_cursor"] is None


def test_date_filter_uses_timezone_and_includes_exact_boundary():
    store = StudioR2Store(SimpleNamespace(list_objects=lambda *args, **kwargs: {"Contents": [{"Key": "uploads/submissions/image.png", "Size": 10, "LastModified": datetime(2026, 10, 1, tzinfo=timezone.utc)}]}), "evidence", ("uploads/submissions/",), 200, 1000)
    assert len(store.list_images("uploads/submissions/", since="2026-10-01T08:00:00+08:00")["objects"]) == 1
    assert store.list_images("uploads/submissions/", since="2026-10-01T08:00:01+08:00")["objects"] == []
    with pytest.raises(ValueError, match="ISO"):
        store.list_images("uploads/submissions/", since="not-a-date")
