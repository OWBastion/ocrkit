from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.roi_config import RoiConfig
from app.ocr.engine import OcrEngine
from app.parser.terminology import TerminologyCatalog
from app.storage.r2_client import R2ObjectStore


@dataclass
class AppContext:
    roi_config: RoiConfig
    map_names: list[str]
    map_aliases: dict[str, str]
    ocr_engine: OcrEngine
    object_store: R2ObjectStore | None = None
    model_version: str = "builtin"
    engine_name: str = "rapidocr"
    layout_version: str = "1280x720-v6"
    roi_variants: tuple[RoiConfig, ...] = ()
    terminology: TerminologyCatalog | None = None


def get_context(request: Request) -> AppContext:
    return request.app.state.ctx
