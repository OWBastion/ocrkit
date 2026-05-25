from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_ocr import router as ocr_router
from app.core.config import settings
from app.core.context import AppContext
from app.core.roi_config import load_map_names, load_roi_config
from app.ocr.engine import OcrEngine
from app.ocr.paddleocr_engine import PaddleOcrEngine
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.storage.r2_client import R2ObjectStore


def _create_ocr_engine() -> OcrEngine:
    if settings.ocr_engine == "paddleocr":
        return PaddleOcrEngine()
    return RapidOcrEngine()


def create_context() -> AppContext:
    object_store = None
    if settings.r2_endpoint_url and settings.r2_access_key_id and settings.r2_secret_access_key:
        object_store = R2ObjectStore.from_settings(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region_name,
            default_bucket=settings.r2_default_bucket,
            allowed_buckets_raw=settings.r2_allowed_buckets,
            read_timeout_seconds=settings.r2_read_timeout_seconds,
        )
    return AppContext(
        roi_config=load_roi_config(settings.roi_config_path),
        map_names=load_map_names(settings.maps_config_path),
        ocr_engine=_create_ocr_engine(),
        object_store=object_store,
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
