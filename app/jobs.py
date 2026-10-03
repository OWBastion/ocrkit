from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.core.context import AppContext
from app.image.loader import decode_image
from app.service import extract_structured

logger = logging.getLogger(__name__)
QUEUE_DEADLINE = 600
RETENTION = 24 * 60 * 60
MAX_PENDING = 100
MAX_RECORDS = 10000


class JobConflict(Exception):
    pass


class JobFull(Exception):
    pass


class JobExpired(Exception):
    pass


def recognize_payload(ctx: AppContext, payload: bytes, job_id: str, debug: bool = False, queued_at: float | None = None):
    with ctx.inference_lock:
        if queued_at is not None and time.time() - queued_at >= QUEUE_DEADLINE:
            raise JobExpired()
        return extract_structured(
            image=decode_image(payload), roi_config=ctx.roi_config,
            map_names=ctx.map_names, map_aliases=ctx.map_aliases, engine=ctx.ocr_engine,
            include_debug=debug, request_id=job_id, engine_name=ctx.engine_name,
            model_version=ctx.model_version, layout_version=ctx.layout_version,
            roi_variants=ctx.roi_variants, terminology=ctx.terminology,
        )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OcrJobs:
    def __init__(self, path: Path, ctx: AppContext, platform_base_url: str, token: str):
        origin = urlsplit(platform_base_url)
        if origin.scheme != "https" or not origin.netloc or origin.username or origin.password or origin.query or origin.fragment or origin.path not in ("", "/"):
            raise ValueError("platform_base_url must be an HTTPS origin")
        self.path, self.ctx = path, ctx
        self.base_url, self.token = platform_base_url.rstrip("/"), token
        self.lock, self.stop = Lock(), Event()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        with self.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, digest TEXT NOT NULL, created REAL NOT NULL, image BLOB, result TEXT, done INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0)")
        os.chmod(path, 0o600)
        self.threads: list[Thread] = []
        self.process_lock = None

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def accept(self, job_id: str, payload: bytes):
        digest = hashlib.sha256(payload).hexdigest()
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM jobs WHERE created < ?", (time.time() - RETENTION,))
            existing = db.execute("SELECT digest FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if existing:
                if existing["digest"] != digest:
                    raise JobConflict()
                return
            counts = db.execute("SELECT COUNT(*) AS total, SUM(done = 0) AS pending FROM jobs").fetchone()
            if counts["total"] >= MAX_RECORDS or (counts["pending"] or 0) >= MAX_PENDING:
                raise JobFull()
            db.execute("INSERT INTO jobs (id, digest, created, image) VALUES (?, ?, ?, ?)", (job_id, digest, time.time(), payload))

    def start(self):
        lock_path = self.path.with_suffix(".lock")
        self.process_lock = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600), "wb")
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.process_lock.close()
            self.process_lock = None
            raise RuntimeError("OCR jobs require one service process per spool") from None
        for worker in (self.recognition_loop, self.delivery_loop):
            thread = Thread(target=worker, daemon=True)
            self.threads.append(thread)
            thread.start()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join()
        if self.process_lock is not None:
            self.process_lock.close()
            self.process_lock = None

    def recognition_loop(self):
        while not self.stop.is_set():
            try:
                if not self.recognize_next():
                    self.stop.wait(0.25)
            except Exception:
                logger.exception("OCR job storage failure")
                self.stop.wait(1)

    def recognize_next(self):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT id, image, created FROM jobs WHERE image IS NOT NULL ORDER BY created LIMIT 1").fetchone()
        if row is None:
            return False
        body = {"contractVersion": "1"}
        if time.time() - row["created"] >= QUEUE_DEADLINE:
            body["errorCode"] = "OCR_JOB_EXPIRED"
        else:
            try:
                body["result"] = recognize_payload(self.ctx, row["image"], row["id"], queued_at=row["created"]).model_dump(mode="json")
            except JobExpired:
                body["errorCode"] = "OCR_JOB_EXPIRED"
            except Exception:
                logger.warning("OCR job recognition failed", exc_info=False)
                body["errorCode"] = "OCR_RECOGNITION_FAILED"
        with self.lock, self.connect() as db:
            db.execute("UPDATE jobs SET image = NULL, result = ? WHERE id = ?", (json.dumps(body), row["id"]))
        return True

    def delivery_loop(self):
        while not self.stop.is_set():
            try:
                if not self.deliver_next():
                    self.stop.wait(0.25)
            except Exception:
                logger.exception("OCR callback storage failure")
                self.stop.wait(1)

    def deliver_next(self):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM jobs WHERE created < ?", (time.time() - RETENTION,))
            row = db.execute("SELECT id, result, attempts FROM jobs WHERE result IS NOT NULL AND next_attempt <= ? ORDER BY created LIMIT 1", (time.time(),)).fetchone()
        if row is None:
            return False
        request = Request(
            f"{self.base_url}/v1/ocrkit/jobs/{row['id']}/result", data=row["result"].encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "User-Agent": "OWBastion-OCRKit/1.0"}, method="POST",
        )
        try:
            with build_opener(NoRedirect()).open(request, timeout=10) as response:
                delivered = 200 <= response.status < 300
        except Exception:
            delivered = False
        with self.lock, self.connect() as db:
            if delivered:
                db.execute("UPDATE jobs SET result = NULL, done = 1 WHERE id = ?", (row["id"],))
            else:
                delay = min(60, 2 ** min(row["attempts"] + 1, 6))
                db.execute("UPDATE jobs SET attempts = attempts + 1, next_attempt = ? WHERE id = ?", (time.time() + delay, row["id"]))
        return True
