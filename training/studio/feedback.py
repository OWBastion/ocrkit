from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import urlparse

from app.core.config import settings

ROOT = Path(__file__).resolve().parents[2]


class ScreenshotFeedback(TypedDict):
    available: bool
    marks: dict[str, Literal["accurate", "inaccurate"]]


def _query(since: str | None) -> str:
    cutoff = ""
    if since:
        parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        cutoff = f" AND a.created_at >= {int(parsed.timestamp() * 1000)}"
    return f"""SELECT a.object_key, f.accuracy
FROM ocr_accuracy_feedback f
JOIN attachments a ON a.submission_id = f.submission_id
WHERE a.upload_status = 'stored' AND a.object_key IS NOT NULL
AND a.id = (SELECT latest.id FROM attachments latest
    WHERE latest.submission_id = f.submission_id AND latest.upload_status = 'stored'
    ORDER BY latest.created_at DESC, latest.id DESC LIMIT 1)
AND f.ocr_result_id = (SELECT r.id FROM ocr_results r
    WHERE r.submission_id = f.submission_id
    ORDER BY r.created_at DESC, r.id DESC LIMIT 1)
{cutoff}
ORDER BY a.created_at DESC, a.id DESC
LIMIT 1000"""


def get_feedback(since: str | None = None) -> ScreenshotFeedback:
    unavailable: ScreenshotFeedback = {"available": False, "marks": {}}
    try:
        platform = ROOT.parent / "owbastion.codes"
        host = urlparse(settings.r2_endpoint_url).hostname or ""
        account = re.fullmatch(r"([a-f0-9]{32})\.r2\.cloudflarestorage\.com", host)
        if account is None or not (platform / "wrangler.toml").is_file():
            return unavailable
        query = _query(since)
        environment = os.environ.copy()
        environment["CLOUDFLARE_ACCOUNT_ID"] = account[1]
        result = subprocess.run(
            ["pnpm", "exec", "wrangler", "d1", "execute", "DB", "--remote", "--json", "--command", query],
            cwd=platform,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return unavailable
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            return unavailable
        response = payload[0]
        if not isinstance(response, dict) or response.get("success") is not True or not isinstance(response.get("results"), list):
            return unavailable
        marks: dict[str, Literal["accurate", "inaccurate"]] = {}
        for row in response["results"]:
            if not isinstance(row, dict) or not isinstance(row.get("object_key"), str) or not row["object_key"] or not isinstance(row.get("accuracy"), str) or row["accuracy"] not in {"accurate", "inaccurate"}:
                return unavailable
            marks[row["object_key"]] = row["accuracy"]
        return {"available": True, "marks": marks}
    except (OSError, subprocess.SubprocessError, ValueError, OverflowError):
        return unavailable
