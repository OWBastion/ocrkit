from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.roi_config import RoiConfig
from app.ocr.engine import OcrEngine


@dataclass
class AppContext:
    roi_config: RoiConfig
    map_names: list[str]
    ocr_engine: OcrEngine


def get_context(request: Request) -> AppContext:
    return request.app.state.ctx
