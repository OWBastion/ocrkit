from __future__ import annotations

from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from app.storage.r2_client import ObjectAccessDeniedError, ObjectNotFoundError, R2ObjectStore


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "operation")


@pytest.fixture
def store() -> R2ObjectStore:
    return R2ObjectStore.from_settings(
        endpoint_url="https://example.invalid",
        access_key_id="key",
        secret_access_key="secret",
        region_name="auto",
        default_bucket="bucket",
        allowed_buckets_raw="",
        read_timeout_seconds=5,
    )


def test_generate_presigned_put_url_targets_the_given_bucket_and_key(store: R2ObjectStore) -> None:
    url = store.generate_presigned_put_url("bucket", "colab-runs/run-1/checkpoint.pdparams", expires_in_seconds=900)

    assert "colab-runs/run-1/checkpoint.pdparams" in url
    assert "X-Amz-Signature" in url


def test_download_object_writes_the_destination_file(store: R2ObjectStore, tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pdparams"

    def fake_download_file(bucket: str, key: str, filename: str) -> None:
        assert (bucket, key) == ("bucket", "checkpoint.pdparams")
        Path(filename).write_bytes(b"weights")

    store._client.download_file = fake_download_file  # type: ignore[attr-defined]

    store.download_object("bucket", "checkpoint.pdparams", destination)

    assert destination.read_bytes() == b"weights"


def test_download_object_missing_key_raises_not_found(store: R2ObjectStore, tmp_path: Path) -> None:
    def fake_download_file(*_args: object) -> None:
        raise _client_error("NoSuchKey")

    store._client.download_file = fake_download_file  # type: ignore[attr-defined]

    with pytest.raises(ObjectNotFoundError):
        store.download_object("bucket", "missing.pdparams", tmp_path / "out")


def test_delete_object_access_denied_raises_access_denied_error(store: R2ObjectStore) -> None:
    def fake_delete_object(**_kwargs: object) -> None:
        raise _client_error("AccessDenied")

    store._client.delete_object = fake_delete_object  # type: ignore[attr-defined]

    with pytest.raises(ObjectAccessDeniedError):
        store.delete_object("bucket", "checkpoint.pdparams")


def test_delete_object_succeeds_without_error(store: R2ObjectStore) -> None:
    calls: list[dict[str, object]] = []
    store._client.delete_object = lambda **kwargs: calls.append(kwargs)  # type: ignore[attr-defined]

    store.delete_object("bucket", "checkpoint.pdparams")

    assert calls == [{"Bucket": "bucket", "Key": "checkpoint.pdparams"}]
