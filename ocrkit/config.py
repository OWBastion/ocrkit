from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    title_source_url: str = os.getenv(
        "TITLE_SOURCE_URL",
        "https://raw.githubusercontent.com/OWBastion/Bastion/main/data/title-source.json",
    )
    title_cache_ttl_sec: int = int(os.getenv("TITLE_CACHE_TTL_SEC", "600"))
    request_timeout_sec: int = int(os.getenv("REQUEST_TIMEOUT_SEC", "20"))


SETTINGS = Settings()
