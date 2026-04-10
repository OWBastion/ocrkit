from __future__ import annotations

import cv2
import numpy as np


ROI_TOP_BAR = [0.10, 0.00, 0.92, 0.18]
ROI_CENTER_BANNER = [0.04, 0.19, 0.96, 0.44]
ROI_LEFT_PANEL = [0.00, 0.12, 0.36, 0.70]


def crop_by_frac(img: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(w * roi[0])
    y1 = int(h * roi[1])
    x2 = int(w * roi[2])
    y2 = int(h * roi[3])
    return img[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)].copy()


def to_white_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 160], np.uint8), np.array([179, 70, 255], np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


class OcrRuntime:
    def __init__(self) -> None:
        self._engines: dict[str, object] = {}

    def warm(self) -> None:
        if self._engines:
            return
        self._engines["ch"] = self._create_engine("ch")
        self._engines["en"] = self._create_engine("en")

    def _create_engine(self, lang: str):
        from paddleocr import PaddleOCR

        return PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            show_log=False,
            use_gpu=False,
        )

    def _ocr_lines(self, image: np.ndarray, lang: str) -> list[str]:
        engine = self._engines[lang]
        res = engine.ocr(image, cls=False)
        if not res:
            return []
        lines: list[str] = []
        for block in res:
            for item in block:
                if not item or len(item) < 2:
                    continue
                text = (item[1][0] or "").strip()
                if text:
                    lines.append(text)
        return lines

    def extract_roi_texts(self, image_bgr: np.ndarray) -> dict[str, str]:
        top_bar = crop_by_frac(image_bgr, ROI_TOP_BAR)
        center = crop_by_frac(image_bgr, ROI_CENTER_BANNER)
        left = crop_by_frac(image_bgr, ROI_LEFT_PANEL)

        top_lines = self._ocr_lines(top_bar, "ch") + self._ocr_lines(to_white_mask(top_bar), "ch")
        center_lines = self._ocr_lines(center, "ch") + self._ocr_lines(to_white_mask(center), "ch")
        left_lines = self._ocr_lines(left, "ch") + self._ocr_lines(to_white_mask(left), "ch")

        return {
            "top_bar": " ".join(top_lines),
            "center_banner": " ".join(center_lines),
            "left_panel": " ".join(left_lines),
        }
