from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from fastapi import FastAPI

from app.api.routes_ocr import router as ocr_router
from app.core.config import settings
from app.core.context import AppContext
from app.catalog import load_agent_title_labels
from app.core.roi_config import load_map_aliases, load_map_names, load_roi_config
from app.ocr.engine import OcrEngine
from app.ocr.paddleocr_engine import PaddleOcrEngine
from app.ocr.rapidocr_engine import RapidOcrEngine
from app.model_artifacts import ModelArtifactStore, load_release_channel
from app.storage.r2_client import R2ObjectStore

logger = logging.getLogger(__name__)


def _create_ocr_engine(model_config_path=None) -> OcrEngine:
    if settings.ocr_engine == "paddleocr":
        return PaddleOcrEngine()
    return RapidOcrEngine(config_path=model_config_path)


def _model_store() -> R2ObjectStore:
    if not (
        settings.r2_endpoint_url
        and settings.r2_access_key_id
        and settings.r2_secret_access_key
        and settings.r2_default_bucket
    ):
        raise RuntimeError("Model R2 settings are incomplete")
    return R2ObjectStore.from_settings(
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region_name,
        default_bucket=settings.r2_default_bucket,
        allowed_buckets_raw=settings.r2_default_bucket,
        read_timeout_seconds=settings.model_download_timeout_seconds,
    )


def _selected_model_manifest(store: R2ObjectStore) -> str:
    if settings.model_release_channel_key:
        return load_release_channel(store, settings.r2_default_bucket, settings.model_release_channel_key).manifest_key
    return settings.model_manifest_key


def _load_model(manifest_key: str) -> tuple[str, object]:
    store = _model_store()
    artifacts = ModelArtifactStore(store, settings.r2_default_bucket, settings.model_cache_dir).prepare(manifest_key)
    return artifacts.version, artifacts.rapidocr_config_path


def create_context() -> AppContext:
    roi_config = load_roi_config(settings.roi_config_path)
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
    model_version = "builtin"
    model_config_path = None
    selected_manifest = ""
    if settings.model_manifest_key or settings.model_release_channel_key:
        store = _model_store()
        selected_manifest = _selected_model_manifest(store)
        model_version, model_config_path = _load_model(selected_manifest)

    return AppContext(
        roi_config=roi_config,
        map_names=load_map_names(settings.maps_config_path),
        map_aliases=load_map_aliases(settings.maps_config_path),
        ocr_engine=_create_ocr_engine(model_config_path),
        object_store=object_store,
        achievement_titles=load_agent_title_labels(
            settings.agents_api_base_url,
            settings.agents_api_timeout_seconds,
        ),
        model_version=model_version,
        model_manifest_key=selected_manifest,
        engine_name=settings.ocr_engine,
        layout_version=roi_config.version,
    )


def _refresh_channel_model(context: AppContext) -> AppContext | None:
    if not settings.model_release_channel_key:
        return None
    store = _model_store()
    manifest_key = _selected_model_manifest(store)
    if manifest_key == context.model_manifest_key:
        return None
    model_version, model_config_path = _load_model(manifest_key)
    return replace(
        context,
        ocr_engine=_create_ocr_engine(model_config_path),
        model_version=model_version,
        model_manifest_key=manifest_key,
    )


async def _watch_model_release_channel(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(max(10, settings.model_refresh_seconds))
        try:
            updated = await asyncio.to_thread(_refresh_channel_model, app.state.ctx)
            if updated is not None:
                app.state.ctx = updated
                logger.info("switched OCR model to release channel version %s", updated.model_version)
        except Exception:
            logger.exception("failed to refresh OCR model release channel; keeping the active model")


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.state.ctx = create_context()
    if settings.model_release_channel_key:
        @app.on_event("startup")
        async def start_model_release_watcher() -> None:
            app.state.model_release_task = asyncio.create_task(_watch_model_release_channel(app))

        @app.on_event("shutdown")
        async def stop_model_release_watcher() -> None:
            task = getattr(app.state, "model_release_task", None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    app.include_router(ocr_router)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "engine": settings.ocr_engine,
            "model_version": app.state.ctx.model_version,
            "application_version": settings.app_version,
            "version": settings.app_version,
        }

    return app


app = create_app()
