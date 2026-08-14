from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCRKit"
    app_version: str = "0.1.0"
    api_token: str = ""
    allow_debug: bool = False
    max_upload_bytes: int = 10 * 1024 * 1024
    ocr_engine: str = "rapidocr"
    roi_config_path: Path = Path("configs/roi_1280x720.yaml")
    maps_config_path: Path = Path("configs/maps.yaml")
    terminology_config_path: Path = Path("configs/terminology.yaml")
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_region_name: str = "auto"
    r2_default_bucket: str = ""
    r2_allowed_buckets: str = ""
    r2_read_timeout_seconds: int = 10
    studio_r2_bucket: str = ""
    studio_r2_allowed_prefixes: str = ""
    studio_r2_max_objects: int = Field(default=200, ge=1, le=1000)
    studio_r2_max_object_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    model_manifest_key: str = ""
    model_release_channel_key: str = ""
    model_cache_dir: Path = Path("/var/lib/ocrkit/models")
    model_download_timeout_seconds: int = 30

    model_config = SettingsConfigDict(env_prefix="OCRKIT_", extra="ignore")


settings = Settings()
