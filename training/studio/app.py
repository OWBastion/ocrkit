from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from training.studio.core import (
    DEFAULT_WORK_ROOT,
    batch_summary,
    create_batch,
    finalize_dataset,
    generate_candidates,
    review_counts,
    review_rows,
    roi_preview_paths,
    update_review_row,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "training/studio/frontend/dist"
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ReviewUpdate(BaseModel):
    split: str
    crop: str
    status: str
    transcription: str | None = None


class TrainingStart(BaseModel):
    resume_checkpoint: str | None = None
    epochs: int = Field(default=10, ge=1, le=100)


def _batch_dir(work_root: Path, batch_id: str) -> Path:
    candidate = (work_root / "batches" / batch_id).resolve()
    if candidate.parent != (work_root / "batches").resolve() or not (candidate / "batch.json").is_file():
        raise HTTPException(status_code=404, detail="batch not found")
    return candidate


def _crop_path(batch_dir: Path, split: str, crop: str) -> Path:
    if split not in {"train", "holdout"}:
        raise HTTPException(status_code=422, detail="invalid split")
    row = next((item for item in review_rows(batch_dir, split) if item.get("crop") == crop), None)
    if row is None:
        raise HTTPException(status_code=404, detail="crop not found")
    path = (batch_dir / "dataset" / crop).resolve()
    dataset = (batch_dir / "dataset").resolve()
    if path.parent != dataset and dataset not in path.parents:
        raise HTTPException(status_code=422, detail="invalid crop path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="crop file not found")
    return path


def _list_batches(work_root: Path) -> list[dict[str, object]]:
    batches_dir = work_root / "batches"
    if not batches_dir.is_dir():
        return []
    summaries: list[dict[str, object]] = []
    for path in sorted(batches_dir.iterdir(), reverse=True):
        if not (path / "batch.json").is_file():
            continue
        summary: dict[str, object] = batch_summary(path)
        summary["review"] = review_counts(path)
        summaries.append(summary)
    return summaries


def _resume_checkpoint(batch_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = (batch_dir / value).resolve()
    runs_dir = (batch_dir / "runs").resolve()
    if runs_dir not in candidate.parents or candidate.suffix:
        raise HTTPException(status_code=422, detail="invalid resume checkpoint")
    if any(not Path(f"{candidate}{suffix}").is_file() for suffix in (".pdparams", ".pdopt", ".states")):
        raise HTTPException(status_code=422, detail="resume checkpoint is incomplete or no longer exists")
    return candidate


def _list_resume_checkpoints(batch_dir: Path) -> list[dict[str, str]]:
    runs_dir = batch_dir / "runs"
    if not runs_dir.is_dir():
        return []
    checkpoints: list[dict[str, str]] = []
    for params in sorted(runs_dir.glob("smoke-*/checkpoints/*.pdparams"), reverse=True):
        checkpoint = params.with_suffix("")
        if all(Path(f"{checkpoint}{suffix}").is_file() for suffix in (".pdopt", ".states")):
            checkpoints.append({"path": checkpoint.relative_to(batch_dir).as_posix(), "name": checkpoint.relative_to(runs_dir).as_posix()})
    return checkpoints


def _poll_training_process(state: dict[str, object]) -> bool:
    """Refresh a Studio-owned training process without mistaking a zombie for a live run."""
    if state.get("status") != "training":
        return False
    pid = int(state["pid"])
    try:
        reaped_pid, wait_status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            state["status"] = "completed_or_failed"
            return True
        return False
    if reaped_pid == 0:
        return False

    exit_code = os.waitstatus_to_exitcode(wait_status)
    state["exit_code"] = exit_code
    state["status"] = "completed" if exit_code == 0 else "failed"
    return True


def create_app(work_root: Path = DEFAULT_WORK_ROOT, frontend_dir: Path = FRONTEND_DIST) -> FastAPI:
    if not (frontend_dir / "index.html").is_file():
        raise RuntimeError(f"Studio frontend is missing: run `pnpm --dir training/studio/frontend build` ({frontend_dir})")

    app = FastAPI(title="OCRKit Studio", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"ok": "true", "service": "ocrkit-studio"}

    @app.get("/api/batches")
    def batches() -> list[dict[str, object]]:
        return _list_batches(work_root)

    @app.post("/api/batches")
    async def import_images(files: list[UploadFile] = File(...), holdout_ratio: float = Form(0.2)) -> dict[str, object]:
        work_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix="studio-import-", dir=work_root))
        try:
            paths: list[Path] = []
            for index, upload in enumerate(files, 1):
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in _IMAGE_SUFFIXES:
                    raise HTTPException(status_code=422, detail=f"unsupported image type: {upload.filename}")
                content = await upload.read(_MAX_UPLOAD_BYTES + 1)
                if len(content) > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"image exceeds {_MAX_UPLOAD_BYTES // 1024 // 1024} MiB: {upload.filename}")
                path = temporary_dir / f"{index:04d}{suffix}"
                path.write_bytes(content)
                paths.append(path)
            batch_dir, summary = await run_in_threadpool(create_batch, paths, work_root=work_root, holdout_ratio=holdout_ratio)
            return {"batch": summary, "previews": roi_preview_paths(batch_dir)}
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @app.post("/api/batches/{batch_id}/candidates")
    async def candidates(batch_id: str) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        summary = await run_in_threadpool(generate_candidates, batch_dir)
        return {"summary": summary, "review": review_counts(batch_dir)}

    @app.get("/api/batches/{batch_id}/review")
    def review(batch_id: str, split: str = "train", status: str = "pending") -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        if split not in {"train", "holdout"} or status not in {"pending", "accepted", "rejected", "all"}:
            raise HTTPException(status_code=422, detail="invalid review filter")
        rows = review_rows(batch_dir, split, status)
        return {"rows": rows, "counts": review_counts(batch_dir)}

    @app.get("/api/batches/{batch_id}/crop")
    def crop(batch_id: str, split: str, crop: str) -> FileResponse:
        return FileResponse(_crop_path(_batch_dir(work_root, batch_id), split, crop))

    @app.put("/api/batches/{batch_id}/review")
    async def save_review(batch_id: str, update: ReviewUpdate) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        row = await run_in_threadpool(update_review_row, batch_dir, update.split, update.crop, update.status, update.transcription)
        return {"row": row, "counts": review_counts(batch_dir)}

    @app.post("/api/batches/{batch_id}/finalize")
    async def finalize(batch_id: str) -> dict[str, int]:
        try:
            return await run_in_threadpool(finalize_dataset, _batch_dir(work_root, batch_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/batches/{batch_id}/training/smoke")
    async def start_smoke(batch_id: str, request: TrainingStart | None = None) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        request = request or TrainingStart()
        try:
            await run_in_threadpool(finalize_dataset, batch_dir)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_dir = batch_dir / "runs" / f"smoke-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training.log"
        command = [
            str(ROOT / "training/run_rec_smoke.sh"),
            "--labels-dir", str(batch_dir / "dataset"),
            "--output-dir", str(run_dir / "checkpoints"),
            "--epochs", str(request.epochs),
        ]
        resume_checkpoint = _resume_checkpoint(batch_dir, request.resume_checkpoint)
        if resume_checkpoint is not None:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state: dict[str, object] = {
            "pid": process.pid,
            "status": "training",
            "command": command,
            "log": str(log_path),
            "epochs": request.epochs,
            "resume_checkpoint": request.resume_checkpoint,
        }
        payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        (run_dir / "run.json").write_text(payload, encoding="utf-8")
        (batch_dir / "runs/latest.json").write_text(payload, encoding="utf-8")
        return state

    @app.get("/api/batches/{batch_id}/training/checkpoints")
    def resume_checkpoints(batch_id: str) -> list[dict[str, str]]:
        return _list_resume_checkpoints(_batch_dir(work_root, batch_id))

    @app.get("/api/batches/{batch_id}/training")
    def training_status(batch_id: str) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        state_path = batch_dir / "runs/latest.json"
        if not state_path.is_file():
            return {"status": "not_started", "log": "", "log_tail": ""}
        state: dict[str, object] = json.loads(state_path.read_text(encoding="utf-8"))
        state_changed = _poll_training_process(state)
        log_path = Path(str(state.get("log", "")))
        # Keep a generous plain-text tail for the studio log viewer.
        state["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-48000:] if log_path.is_file() else ""
        if state_changed:
            # Persist terminal status without embedding the log body into latest.json.
            persisted = {key: value for key, value in state.items() if key != "log_tail"}
            state_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return state

    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="studio")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only OCRKit Studio API and Vite-built frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--frontend-dir", type=Path, default=FRONTEND_DIST)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.work_root, args.frontend_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
