from __future__ import annotations

from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, HTTPException

from .config import SETTINGS
from .models import ExtractRequest, ExtractResponse, PingResponse
from .ocr import OcrRuntime
from .rules import TitleRulesStore
from .service import extract_from_image_url


def create_app() -> FastAPI:
    rules_store = TitleRulesStore()
    ocr_runtime = OcrRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = aiohttp.ClientTimeout(total=SETTINGS.request_timeout_sec)
        app.state.http = aiohttp.ClientSession(timeout=timeout)
        app.state.rules_store = rules_store
        app.state.ocr_runtime = ocr_runtime
        app.state.ocr_runtime.warm()
        await app.state.rules_store.get_rules(app.state.http)
        yield
        await app.state.http.close()

    app = FastAPI(title="Bastion OCR", version="0.1.0", lifespan=lifespan)

    @app.get("/ping", response_model=PingResponse)
    async def ping() -> PingResponse:
        rules = app.state.rules_store._rules
        return PingResponse(
            ok=True,
            rules_version=(rules.version if rules else None),
            loaded_at_epoch=(rules.loaded_at_epoch if rules else None),
        )

    @app.post("/extract", response_model=ExtractResponse)
    async def extract(payload: ExtractRequest) -> ExtractResponse:
        try:
            return await extract_from_image_url(
                session=app.state.http,
                rules_store=app.state.rules_store,
                ocr_runtime=app.state.ocr_runtime,
                image_url=str(payload.image_url),
            )
        except aiohttp.ClientResponseError as e:
            raise HTTPException(status_code=400, detail=f"image fetch failed: {e.status}") from e
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"image fetch failed: {e}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"extract failed: {e}") from e

    return app
