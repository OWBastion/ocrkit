from __future__ import annotations

from typing import Any

from app.core.roi_config import RoiConfig
from app.image.preprocess import preprocess_by_roi
from app.image.quality import assess_input_quality
from app.image.roi import crop_all_rois, select_roi_config
from app.ocr.engine import OcrEngine
from app.parser.bottom_left_hero import parse_bottom_left_hero
from app.parser.center_summary import parse_center_summary
from app.parser.left_panel import parse_achievement_titles, parse_left_panel
from app.parser.result_merger import merge_result
from app.parser.right_panel import parse_right_panel
from app.parser.run_code import ParsedRunCode, enforce_run_code_confidence, parse_run_code
from app.schemas.response import ChallengeResponse, DebugPayload, FieldEvidence, QualityPayload


_FIELD_ROIS = {
    "challenge_completed": ("left_panel", "center_banner"),
    "heroes_completed": ("left_panel",),
    "heroes_total": ("left_panel",),
    "viewer_player": ("bottom_left_hero",),
    "achievement_title": ("achievement_panel",),
    "achievement_titles": ("achievement_panel",),
    "achievement_unlocked": ("achievement_panel",),
    "achievement_panel_text": ("achievement_panel",),
    "deaths": ("left_panel", "center_banner"),
    "skips": ("left_panel", "center_banner"),
    "duration_text": ("left_panel", "center_banner"),
    "duration_seconds": ("left_panel", "center_banner"),
    "map_name": ("right_panel",),
    "map_variant": ("right_panel",),
    "difficulty": ("right_panel",),
    "version": ("right_panel",),
    "run_code": ("run_code_panel",),
}
_RUN_CODE_ROIS = ("run_code_panel", "run_code_right_panel")


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


def _build_field_evidence(
    data,
    confidences: dict[str, float],
    run_code: ParsedRunCode,
    run_code_rois: tuple[str, ...],
) -> dict[str, FieldEvidence]:
    fields = {
        name: _field_evidence(name, getattr(data, name), confidences)
        for name in _FIELD_ROIS
    }
    run_code_confidence = max((confidences.get(roi, 0.0) for roi in run_code_rois), default=0.0)
    fields["run_code"] = FieldEvidence(
        value=data.run_code,
        confidence=run_code_confidence if run_code.status != "missing" else 0.0,
        source_roi=list(run_code_rois),
        normalization=list(run_code.normalization),
        status=run_code.status,
    )
    return fields


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
    roi_variants: tuple[RoiConfig, ...] = (),
) -> ChallengeResponse:
    active_roi_config = select_roi_config(image, (roi_config, *roi_variants))
    input_quality = assess_input_quality(image, active_roi_config.width, active_roi_config.height)
    normalized, roi_images = crop_all_rois(image, active_roi_config)

    raw_text: dict[str, str] = {}
    confidences: dict[str, float] = {}

    for roi_name, roi_image in roi_images.items():
        processed = preprocess_by_roi(roi_name, roi_image)
        result = engine.recognize(processed)
        raw_text[roi_name] = result.text
        confidences[roi_name] = result.confidence

    center = parse_center_summary(raw_text.get("center_banner", ""))
    left_text = raw_text.get("left_panel", "")
    achievement_text = raw_text.get("achievement_panel", "")
    left = parse_left_panel(left_text)
    achievement_titles = parse_achievement_titles(achievement_text)
    if achievement_titles:
        left.achievement_title = achievement_titles[0]
        left.achievement_titles = achievement_titles
        left.achievement_unlocked = True
    run_code_rois = tuple(
        name for name in _RUN_CODE_ROIS if name in roi_images and raw_text.get(name, "").strip()
    )
    evidence_run_code_rois = run_code_rois or tuple(name for name in _RUN_CODE_ROIS if name in roi_images)[:1]
    run_code = enforce_run_code_confidence(
        parse_run_code("\n".join(raw_text.get(name, "") for name in run_code_rois)),
        max((confidences.get(name, 0.0) for name in run_code_rois), default=0.0),
    )
    bottom_left = parse_bottom_left_hero(raw_text.get("bottom_left_hero", ""))
    right = parse_right_panel(raw_text.get("right_panel", ""), map_names, map_aliases)
    data = merge_result(
        center,
        left,
        bottom_left,
        right,
        achievement_panel_text=achievement_text.strip() or None,
        run_code=run_code.value,
    )

    warnings: list[str] = list(input_quality["warnings"])
    if data.heroes_completed is None or data.heroes_total is None:
        warnings.append("left_panel.hero_progress_missing")
    if data.deaths is None or data.skips is None:
        warnings.append("left_panel.deaths_skips_missing")
    if data.version is None:
        warnings.append("right_panel.version_missing")
    if run_code.warning is not None:
        warnings.append(run_code.warning)
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
                for name, box in active_roi_config.rois.items()
            },
            raw_text=raw_text,
            confidence=confidences,
        )

    return ChallengeResponse(
        request_id=request_id,
        engine=engine_name,
        model_version=model_version,
        layout_version=active_roi_config.version,
        ok=True,
        data=data,
        fields=_build_field_evidence(data, confidences, run_code, evidence_run_code_rois),
        warnings=warnings,
        quality=QualityPayload(
            original_size=input_quality["original_size"],
            aspect_ratio=input_quality["aspect_ratio"],
            layout_confidence=input_quality["layout_confidence"],
            cropped=input_quality["cropped"],
            blur_score=input_quality["blur_score"],
            normalized_size=(normalized.shape[1], normalized.shape[0]),
            layout_version=active_roi_config.version,
            warnings=warnings,
        ),
        debug=debug_payload,
    )
