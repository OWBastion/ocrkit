from __future__ import annotations

from app.core.roi_config import RoiConfig
from app.image.preprocess import preprocess_by_roi
from app.image.roi import crop_all_rois
from app.ocr.engine import OcrEngine
from app.parser.center_summary import parse_center_summary
from app.parser.left_panel import parse_left_panel
from app.parser.result_merger import merge_result
from app.parser.right_panel import parse_right_panel
from app.schemas.response import ChallengeResponse, DebugPayload


def extract_structured(
    image,
    roi_config: RoiConfig,
    map_names: list[str],
    engine: OcrEngine,
    include_debug: bool,
) -> ChallengeResponse:
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
    right = parse_right_panel(raw_text.get("right_panel", ""), map_names)
    data = merge_result(center, left, right)

    warnings: list[str] = []
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

    return ChallengeResponse(ok=True, data=data, warnings=warnings, debug=debug_payload)
