from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.context import AppContext, get_context
from app.core.errors import ErrorBody, ErrorResponse
from app.image.loader import SUPPORTED_MIME, decode_image
from app.schemas.response import ChallengeResponse
from app.service import extract_structured
from app.storage.r2_client import (
    ObjectAccessDeniedError,
    ObjectDownloadError,
    ObjectNotFoundError,
    ObjectTimeoutError,
)

router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


class ChallengeByObjectRequest(BaseModel):
    object_key: str = Field(..., min_length=1, max_length=1024)
    bucket: str | None = Field(default=None, max_length=128)
    version_id: str | None = Field(default=None, max_length=256)
    debug: bool = False


def _validate_object_key(object_key: str) -> None:
    key = object_key.strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="INVALID_OBJECT_KEY", message="object_key is required").model_dump(),
        )
    if key.startswith("/") or key.startswith("../") or "/../" in key:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="INVALID_OBJECT_KEY", message="object_key has invalid prefix").model_dump(),
        )


@router.post("/challenge", response_model=ChallengeResponse, responses={400: {"model": ErrorResponse}})
async def recognize_challenge(
    file: UploadFile = File(...),
    debug: bool = Query(default=False),
    ctx: AppContext = Depends(get_context),
) -> ChallengeResponse:
    if file.content_type not in SUPPORTED_MIME:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="INVALID_IMAGE", message="Unsupported content type").model_dump(),
        )

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="IMAGE_TOO_LARGE", message="Image exceeds upload limit").model_dump(),
        )

    try:
        image = decode_image(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="INVALID_IMAGE", message=str(exc)).model_dump(),
        ) from exc

    return extract_structured(
        image=image,
        roi_config=ctx.roi_config,
        map_names=ctx.map_names,
        engine=ctx.ocr_engine,
        include_debug=debug,
    )


@router.post("/challenge/by-object", response_model=ChallengeResponse, responses={400: {"model": ErrorResponse}})
async def recognize_challenge_by_object(
    req: ChallengeByObjectRequest,
    ctx: AppContext = Depends(get_context),
) -> ChallengeResponse:
    if ctx.object_store is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorBody(code="OBJECT_STORE_UNAVAILABLE", message="Object store is not configured").model_dump(),
        )

    _validate_object_key(req.object_key)

    try:
        bucket = ctx.object_store.resolve_bucket(req.bucket)
        payload = ctx.object_store.get_object_bytes(
            bucket=bucket,
            object_key=req.object_key.strip(),
            version_id=req.version_id,
        )
    except ObjectNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(code="OBJECT_NOT_FOUND", message=str(exc)).model_dump(),
        ) from exc
    except ObjectAccessDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=ErrorBody(code="OBJECT_ACCESS_DENIED", message=str(exc)).model_dump(),
        ) from exc
    except ObjectTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=ErrorBody(code="OBJECT_DOWNLOAD_TIMEOUT", message=str(exc)).model_dump(),
        ) from exc
    except ObjectDownloadError as exc:
        raise HTTPException(
            status_code=502,
            detail=ErrorBody(code="OBJECT_DOWNLOAD_FAILED", message=str(exc)).model_dump(),
        ) from exc

    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="IMAGE_TOO_LARGE", message="Image exceeds upload limit").model_dump(),
        )

    try:
        image = decode_image(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=ErrorBody(code="INVALID_IMAGE", message=str(exc)).model_dump(),
        ) from exc

    return extract_structured(
        image=image,
        roi_config=ctx.roi_config,
        map_names=ctx.map_names,
        engine=ctx.ocr_engine,
        include_debug=req.debug,
    )
