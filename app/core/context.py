from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.roi_config import RoiConfig
from app.ocr.engine import OcrEngine
from app.storage.r2_client import R2ObjectStore


@dataclass
class AppContext:
    roi_config: RoiConfig
    map_names: list[str]
    map_aliases: dict[str, str]
    ocr_engine: OcrEngine
    object_store: R2ObjectStore | None = None


def get_context(request: Request) -> AppContext:
    return request.app.state.ctx
