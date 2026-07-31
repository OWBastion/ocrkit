from __future__ import annotations

from dataclasses import dataclass
from socket import timeout as SocketTimeout

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError


class ObjectNotFoundError(Exception):
    pass


class ObjectAccessDeniedError(Exception):
    pass


class ObjectTimeoutError(Exception):
    pass


class ObjectDownloadError(Exception):
    pass


class ObjectListError(Exception):
    pass


class ObjectTooLargeError(Exception):
    pass


@dataclass
class R2ObjectStore:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region_name: str
    default_bucket: str
    allowed_buckets: set[str]
    read_timeout_seconds: int

    def __post_init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
            config=Config(read_timeout=self.read_timeout_seconds, connect_timeout=self.read_timeout_seconds),
        )

    @classmethod
    def from_settings(
        cls,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str,
        default_bucket: str,
        allowed_buckets_raw: str,
        read_timeout_seconds: int,
    ) -> "R2ObjectStore":
        allowed_buckets = {b.strip() for b in allowed_buckets_raw.split(",") if b.strip()}
        if default_bucket:
            allowed_buckets.add(default_bucket)
        return cls(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name=region_name,
            default_bucket=default_bucket,
            allowed_buckets=allowed_buckets,
            read_timeout_seconds=read_timeout_seconds,
        )

    def resolve_bucket(self, bucket: str | None) -> str:
        selected = bucket or self.default_bucket
        if not selected:
            raise ObjectAccessDeniedError("No bucket configured")
        if self.allowed_buckets and selected not in self.allowed_buckets:
            raise ObjectAccessDeniedError("Bucket is not allowed")
        return selected

    def list_objects(
        self,
        bucket: str,
        prefix: str,
        continuation_token: str | None = None,
        max_keys: int = 100,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        try:
            response = self._client.list_objects_v2(**kwargs)
            return response
        except (ConnectTimeoutError, ReadTimeoutError, SocketTimeout) as exc:
            raise ObjectTimeoutError("Object listing timed out") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"AccessDenied", "403"}:
                raise ObjectAccessDeniedError("Object listing access denied") from exc
            raise ObjectListError("Object listing failed") from exc

    def get_object_bytes(
        self,
        bucket: str,
        object_key: str,
        version_id: str | None = None,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        kwargs: dict[str, str] = {"Bucket": bucket, "Key": object_key}
        if version_id:
            kwargs["VersionId"] = version_id
        try:
            response = self._client.get_object(**kwargs)
            body = response["Body"]
            try:
                content = body.read(max_bytes + 1) if max_bytes is not None else body.read()
            finally:
                body.close()
            if max_bytes is not None and len(content) > max_bytes:
                raise ObjectTooLargeError("Object exceeds the configured size limit")
            return content
        except (ConnectTimeoutError, ReadTimeoutError, SocketTimeout) as exc:
            raise ObjectTimeoutError("Object download timed out") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "NoSuchBucket", "404"}:
                raise ObjectNotFoundError("Object not found") from exc
            if code in {"AccessDenied", "403"}:
                raise ObjectAccessDeniedError("Object access denied") from exc
            raise ObjectDownloadError("Object download failed") from exc
