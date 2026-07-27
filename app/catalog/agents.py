from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def load_agent_title_labels(base_url: str, timeout_seconds: int) -> tuple[str, ...]:
    if not base_url.strip():
        return ()

    labels: list[str] = []
    page = 1
    while True:
        query = urlencode({"page": page, "pageSize": 100})
        request = Request(f"{base_url.rstrip('/')}/v1/agents/titles?{query}")
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - configured platform endpoint
            payload = json.load(response)
        labels.extend(
            str(item["label"])
            for item in payload.get("items", [])
            if isinstance(item, dict) and str(item.get("label", "")).strip()
        )
        if not payload.get("hasMore"):
            break
        page += 1
    return tuple(dict.fromkeys(labels))
