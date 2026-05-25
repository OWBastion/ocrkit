from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.core.context import AppContext, get_context
from app.core.errors import ErrorBody, ErrorResponse
from app.image.loader import SUPPORTED_MIME, decode_image
from app.schemas.response import ChallengeResponse
from app.service import extract_structured


router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


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
