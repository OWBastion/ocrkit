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
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_region_name: str = "auto"
    r2_default_bucket: str = ""
    r2_allowed_buckets: str = ""
    r2_read_timeout_seconds: int = 10
    model_manifest_key: str = ""
    model_release_channel_key: str = ""
    model_cache_dir: Path = Path("/var/lib/ocrkit/models")
    model_download_timeout_seconds: int = 30
    agents_api_base_url: str = ""
    agents_api_timeout_seconds: int = 5

    model_config = SettingsConfigDict(env_prefix="OCRKIT_", extra="ignore")


settings = Settings()
