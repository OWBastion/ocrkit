from __future__ import annotations

import re

_AI_MARK = re.compile(r"AI|人工智能", re.IGNORECASE)


def is_ai_mark(text: str) -> bool:
    """Whether OCR text from the watermark region reads as an AI-generation mark such as "豆包AI生成"."""
    return "生成" in text and bool(_AI_MARK.search(text))
