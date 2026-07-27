from __future__ import annotations

from typing import Any

from app.core.roi_config import RoiConfig
from app.image.preprocess import preprocess_by_roi
from app.image.quality import assess_input_quality
from app.image.roi import crop_all_rois
from app.ocr.engine import OcrEngine
from app.parser.bottom_left_hero import parse_bottom_left_hero
from app.parser.center_summary import parse_center_summary
from app.parser.left_panel import parse_left_panel
from app.parser.result_merger import merge_result
from app.parser.right_panel import parse_right_panel
from app.schemas.response import ChallengeResponse, DebugPayload, FieldEvidence, QualityPayload


_FIELD_ROIS = {
    "challenge_completed": ("left_panel", "center_banner"),
    "heroes_completed": ("left_panel",),
    "heroes_total": ("left_panel",),
    "viewer_player": ("bottom_left_hero",),
    "achievement_title": ("bottom_left_hero",),
    "achievement_unlocked": ("bottom_left_hero",),
    "deaths": ("left_panel", "center_banner"),
    "skips": ("left_panel", "center_banner"),
    "duration_text": ("left_panel", "center_banner"),
    "duration_seconds": ("left_panel", "center_banner"),
    "map_name": ("right_panel",),
    "difficulty": ("right_panel",),
    "version": ("right_panel",),
}


def _field_evidence(name: str, value: Any, confidences: dict[str, float]) -> FieldEvidence:
    source_rois = list(_FIELD_ROIS[name])
    available_rois = [roi for roi in source_rois if confidences.get(roi, 0.0) > 0]
    confidence = max((confidences.get(roi, 0.0) for roi in source_rois), default=0.0)
    return FieldEvidence(
        value=value,
        confidence=confidence if value is not None else 0.0,
        source_roi=available_rois if value is not None else source_rois,
        status="ok" if value is not None else "missing",
    )


def _build_field_evidence(data, confidences: dict[str, float]) -> dict[str, FieldEvidence]:
    return {
        name: _field_evidence(name, getattr(data, name), confidences)
        for name in _FIELD_ROIS
    }


def extract_structured(
    image,
    roi_config: RoiConfig,
    map_names: list[str],
    map_aliases: dict[str, str],
    engine: OcrEngine,
    include_debug: bool,
    request_id: str,
    engine_name: str,
    model_version: str,
    layout_version: str,
) -> ChallengeResponse:
    input_quality = assess_input_quality(image, roi_config.width, roi_config.height)
    normalized, roi_images = crop_all_rois(image, roi_config)

    raw_text: dict[str, str] = {}
    confidences: dict[str, float] = {}

    for roi_name, roi_image in roi_images.items():
        processed = preprocess_by_roi(roi_name, roi_image)
        result = engine.recognize(processed)
        raw_text[roi_name] = result.text
        confidences[roi_name] = result.confidence

    center = parse_center_summary(raw_text.get("center_banner", ""))
    left = parse_left_panel(raw_text.get("left_panel", ""))
    bottom_left = parse_bottom_left_hero(raw_text.get("bottom_left_hero", ""))
    right = parse_right_panel(raw_text.get("right_panel", ""), map_names, map_aliases)
    data = merge_result(
        center,
        left,
        bottom_left,
        right,
    )

    warnings: list[str] = list(input_quality["warnings"])
    if data.heroes_completed is None or data.heroes_total is None:
        warnings.append("left_panel.hero_progress_missing")
    if data.deaths is None or data.skips is None:
        warnings.append("left_panel.deaths_skips_missing")
    if data.version is None:
        warnings.append("right_panel.version_missing")
    debug_payload = None
    if include_debug:
        debug_payload = DebugPayload(
            normalized_size=(normalized.shape[1], normalized.shape[0]),
            roi_coordinates={
                name: {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                }
                for name, box in roi_config.rois.items()
            },
            raw_text=raw_text,
            confidence=confidences,
        )

    return ChallengeResponse(
        request_id=request_id,
        engine=engine_name,
        model_version=model_version,
        layout_version=layout_version,
        ok=True,
        data=data,
        fields=_build_field_evidence(data, confidences),
        warnings=warnings,
        quality=QualityPayload(
            original_size=input_quality["original_size"],
            aspect_ratio=input_quality["aspect_ratio"],
            layout_confidence=input_quality["layout_confidence"],
            cropped=input_quality["cropped"],
            blur_score=input_quality["blur_score"],
            normalized_size=(normalized.shape[1], normalized.shape[0]),
            layout_version=layout_version,
            warnings=warnings,
        ),
        debug=debug_payload,
    )
