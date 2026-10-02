from __future__ import annotations

import json
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from training.studio import feedback


def prepare_platform(monkeypatch, tmp_path):
    root = tmp_path / "ocrkit"
    root.mkdir()
    platform = tmp_path / "owbastion.codes"
    platform.mkdir()
    (platform / "wrangler.toml").write_text('name = "local-test"')
    monkeypatch.setattr(feedback, "ROOT", root)
    monkeypatch.setattr(feedback, "settings", SimpleNamespace(r2_endpoint_url="https://" + "a" * 32 + ".r2.cloudflarestorage.com"))
    return platform


def test_feedback_uses_only_current_ocr_and_latest_stored_evidence(monkeypatch, tmp_path):
    platform = prepare_platform(monkeypatch, tmp_path)
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript("""
        CREATE TABLE attachments (id TEXT, submission_id TEXT, object_key TEXT, upload_status TEXT, created_at INTEGER);
        CREATE TABLE ocr_results (id TEXT, submission_id TEXT, created_at INTEGER);
        CREATE TABLE ocr_accuracy_feedback (submission_id TEXT, ocr_result_id TEXT, accuracy TEXT);
        INSERT INTO attachments VALUES
          ('a-old','current','uploads/submissions/current/old.png','stored',1000),
          ('a-new','current','uploads/submissions/current/new.png','stored',2000),
          ('a-pending','current','uploads/submissions/current/pending.png','pending',3000),
          ('a-stale','stale','uploads/submissions/stale/image.png','stored',2000),
          ('a-missing','missing',NULL,'stored',2000),
          ('a-early','early','uploads/submissions/early/image.png','stored',500);
        INSERT INTO ocr_results VALUES
          ('r-old','current',1000), ('r-new','current',2000),
          ('r-stale','stale',1000), ('r-latest','stale',2000),
          ('r-missing','missing',2000), ('r-early','early',500);
        INSERT INTO ocr_accuracy_feedback VALUES
          ('current','r-old','accurate'), ('current','r-new','inaccurate'),
          ('stale','r-stale','inaccurate'), ('missing','r-missing','accurate'),
          ('early','r-early','accurate');
    """)
    def run(command, **kwargs):
        assert command[:9] == ["pnpm", "exec", "wrangler", "d1", "execute", "DB", "--remote", "--json", "--command"]
        assert kwargs["cwd"] == platform
        assert kwargs["env"]["CLOUDFLARE_ACCOUNT_ID"] == "a" * 32
        assert kwargs["timeout"] == 20
        rows = [dict(row) for row in database.execute(command[-1])]
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"success": True, "results": rows}]))
    monkeypatch.setattr(feedback.subprocess, "run", run)
    assert feedback.get_feedback("1970-01-01T00:00:01Z") == {
        "available": True, "marks": {"uploads/submissions/current/new.png": "inaccurate"},
    }
    assert feedback.get_feedback()["marks"] == {
        "uploads/submissions/current/new.png": "inaccurate",
        "uploads/submissions/early/image.png": "accurate",
    }
    database.close()


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("wrangler", 20), OSError("pnpm unavailable"), SimpleNamespace(returncode=1, stdout="private authentication response")])
def test_optional_feedback_authentication_or_transport_failure_never_blocks_r2(monkeypatch, tmp_path, failure):
    prepare_platform(monkeypatch, tmp_path)
    def run(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(feedback.subprocess, "run", run)
    assert feedback.get_feedback() == {"available": False, "marks": {}}


@pytest.mark.parametrize("payload", ["not json", {}, [], [{"success": False, "results": []}], [{"success": True, "results": [{"object_key": "image.png", "accuracy": "invented"}]}], [{"success": True, "results": [{"object_key": None, "accuracy": "accurate"}]}], [{"success": True, "results": [{"object_key": "image.png", "accuracy": []}]}]])
def test_invalid_feedback_output_returns_no_marks(monkeypatch, tmp_path, payload):
    prepare_platform(monkeypatch, tmp_path)
    monkeypatch.setattr(feedback.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=payload if isinstance(payload, str) else json.dumps(payload)))
    assert feedback.get_feedback() == {"available": False, "marks": {}}


def test_invalid_date_or_missing_local_platform_does_not_run_cli(monkeypatch, tmp_path):
    platform = prepare_platform(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(feedback.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    assert feedback.get_feedback("';DROP TABLE attachments;") == {"available": False, "marks": {}}
    monkeypatch.setattr(feedback, "ROOT", platform / "missing" / "ocrkit")
    assert feedback.get_feedback() == {"available": False, "marks": {}}
    assert calls == []
