from __future__ import annotations

from io import BytesIO

import aiohttp
import cv2
import numpy as np
from PIL import Image

from .models import ExtractResponse, Extracted, OcrTexts
from .ocr import OcrRuntime
from .parsers import parse_run_from_texts
from .rules import TitleRulesStore, evaluate_titles


async def fetch_image_bgr(session: aiohttp.ClientSession, image_url: str) -> np.ndarray:
    async with session.get(image_url) as resp:
        resp.raise_for_status()
        payload = await resp.read()
    image = Image.open(BytesIO(payload)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


async def extract_from_image_url(
    session: aiohttp.ClientSession,
    rules_store: TitleRulesStore,
    ocr_runtime: OcrRuntime,
    image_url: str,
) -> ExtractResponse:
    rules = await rules_store.get_rules(session)
    image = await fetch_image_bgr(session, image_url)
    texts = ocr_runtime.extract_roi_texts(image)
    parsed = parse_run_from_texts(
        center_banner=texts["center_banner"],
        top_bar=texts["top_bar"],
        left_panel=texts["left_panel"],
        map_labels=rules.map_labels,
    )

    extracted = Extracted(
        passed=parsed.passed,
        player_name=parsed.player_name,
        time_sec=parsed.time_sec,
        deaths=parsed.deaths,
        skips=parsed.skips,
        map_label=parsed.map_label,
        difficulty=parsed.difficulty,
        ocr_texts=OcrTexts(**texts),
    )
    title_decision = evaluate_titles(extracted, rules)
    return ExtractResponse(extracted=extracted, title_decision=title_decision)
