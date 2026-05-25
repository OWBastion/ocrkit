from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCRKit"
    app_version: str = "0.1.0"
    max_upload_bytes: int = 10 * 1024 * 1024
    ocr_engine: str = "rapidocr"
    roi_config_path: Path = Path("configs/roi_1280x720.yaml")
    maps_config_path: Path = Path("configs/maps.yaml")

    model_config = SettingsConfigDict(env_prefix="OCRKIT_", extra="ignore")


settings = Settings()
