from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_ocr import router as ocr_router
from app.core.config import settings
from app.core.context import AppContext
from app.core.roi_config import load_map_names, load_roi_config
from app.ocr.engine import OcrEngine
from app.ocr.paddleocr_engine import PaddleOcrEngine
from app.ocr.rapidocr_engine import RapidOcrEngine


def _create_ocr_engine() -> OcrEngine:
    if settings.ocr_engine == "paddleocr":
        return PaddleOcrEngine()
    return RapidOcrEngine()


def create_context() -> AppContext:
    return AppContext(
        roi_config=load_roi_config(settings.roi_config_path),
        map_names=load_map_names(settings.maps_config_path),
        ocr_engine=_create_ocr_engine(),
    )


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.state.ctx = create_context()
    app.include_router(ocr_router)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "engine": settings.ocr_engine,
            "version": settings.app_version,
        }

    return app


app = create_app()
