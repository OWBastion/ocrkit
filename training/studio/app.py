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
    append_sources,
    batch_summary,
    create_batch,
    export_dataset,
    finalize_dataset,
    generate_candidates,
    review_counts,
    review_rows,
    roi_preview_paths,
    update_review_row,
)
from training.studio.r2 import StudioR2Store, r2_error_detail

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


class PublishStart(BaseModel):
    confirmed: bool = False


class RemoteSourceSelection(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=200)
    holdout_ratio: float = Field(default=0.2, ge=0, lt=1)


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


def _legacy_checkpoint_root(work_root: Path) -> Path:
    return (work_root.parent / "checkpoints").resolve()


def _has_complete_checkpoint(checkpoint: Path) -> bool:
    return all(
        Path(f"{checkpoint}{suffix}").is_file() for suffix in (".pdparams", ".pdopt", ".states")
    )


def _resume_checkpoint(work_root: Path, batch_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None

    if value.startswith("legacy:"):
        legacy_root = _legacy_checkpoint_root(work_root)
        relative = value.removeprefix("legacy:")
        candidate = (legacy_root / relative).resolve()
        if legacy_root not in candidate.parents or candidate.suffix:
            raise HTTPException(status_code=422, detail="invalid legacy resume checkpoint")
        if not _has_complete_checkpoint(candidate):
            raise HTTPException(status_code=422, detail="legacy resume checkpoint is incomplete or no longer exists")
        return candidate

    owner = batch_dir
    relative = value
    if ":" in value:
        owner_id, relative = value.split(":", 1)
        owner = _batch_dir(work_root, owner_id)
    candidate = (owner / relative).resolve()
    runs_dir = (owner / "runs").resolve()
    if runs_dir not in candidate.parents or candidate.suffix:
        raise HTTPException(status_code=422, detail="invalid resume checkpoint")
    if not _has_complete_checkpoint(candidate):
        raise HTTPException(status_code=422, detail="resume checkpoint is incomplete or no longer exists")
    return candidate


def _list_resume_checkpoints(work_root: Path, batch_dir: Path) -> list[dict[str, str]]:
    batches_dir = work_root / "batches"
    checkpoints: list[dict[str, str]] = []
    if batches_dir.is_dir():
        for owner in sorted(batches_dir.iterdir(), key=lambda path: (path != batch_dir, path.name), reverse=False):
            if not (owner / "batch.json").is_file():
                continue
            owner_id = owner.name
            runs_dir = owner / "runs"
            if not runs_dir.is_dir():
                continue
            for params in sorted(runs_dir.glob("smoke-*/checkpoints/*.pdparams"), reverse=True):
                checkpoint = params.with_suffix("")
                if _has_complete_checkpoint(checkpoint):
                    relative = checkpoint.relative_to(owner).as_posix()
                    checkpoints.append({
                        "path": f"{owner_id}:{relative}",
                        "name": f"{owner_id} · {checkpoint.relative_to(runs_dir).as_posix()}",
                    })

    legacy_root = _legacy_checkpoint_root(work_root)
    if legacy_root.is_dir():
        for params in sorted(legacy_root.glob("*/best_accuracy.pdparams"), reverse=True):
            checkpoint = params.with_suffix("")
            if _has_complete_checkpoint(checkpoint):
                relative = checkpoint.relative_to(legacy_root).as_posix()
                checkpoints.append({
                    "path": f"legacy:{relative}",
                    "name": f"历史模型 · {relative}",
                })
    return checkpoints


async def _uploaded_paths(files: list[UploadFile], temporary_dir: Path) -> list[Path]:
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
    return paths


def _poll_training_process(state: dict[str, object], active_status: str = "training") -> bool:
    """Refresh a Studio-owned training process without mistaking a zombie for a live run."""
    if state.get("status") != active_status:
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


def _checkpoint_from_training_state(batch_dir: Path) -> Path:
    state_path = batch_dir / "runs/latest.json"
    if not state_path.is_file():
        raise HTTPException(status_code=422, detail="complete a successful Smoke training run before publishing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("exit_code") != 0:
        raise HTTPException(status_code=422, detail="latest Smoke training run has not passed")
    checkpoint = Path(str(state["log"])).parent / "checkpoints/best_accuracy"
    if not checkpoint.with_suffix(".pdparams").is_file():
        raise HTTPException(status_code=422, detail="latest Smoke run has no best_accuracy checkpoint")
    return checkpoint


def create_app(
    work_root: Path = DEFAULT_WORK_ROOT,
    frontend_dir: Path | None = FRONTEND_DIST,
    remote_store: StudioR2Store | None = None,
) -> FastAPI:
    if frontend_dir is not None and not (frontend_dir / "index.html").is_file():
        raise RuntimeError(f"Studio frontend is missing: run `pnpm --dir training/studio/frontend build` ({frontend_dir})")

    app = FastAPI(title="OCRKit Studio", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"ok": "true", "service": "ocrkit-studio"}

    @app.get("/api/batches")
    def batches() -> list[dict[str, object]]:
        return _list_batches(work_root)

    @app.get("/api/r2/status")
    def r2_status() -> dict[str, object]:
        store = remote_store or StudioR2Store.from_settings()
        if store is None:
            return {"configured": False, "bucket": "", "allowed_prefixes": []}
        return {
            "configured": True,
            "bucket": store.bucket,
            "allowed_prefixes": list(store.allowed_prefixes),
            "max_objects": store.max_objects,
            "max_object_bytes": store.max_object_bytes,
        }

    @app.get("/api/r2/images")
    def r2_images(prefix: str = "", cursor: str | None = None) -> dict[str, object]:
        store = remote_store or StudioR2Store.from_settings()
        if store is None:
            raise HTTPException(status_code=503, detail="Studio R2 未配置")
        selected_prefix = prefix or store.allowed_prefixes[0]
        try:
            return store.list_images(selected_prefix, cursor)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=r2_error_detail(exc)) from exc

    def download_remote_sources(selection: RemoteSourceSelection, temporary_dir: Path) -> tuple[list[Path], dict[str, dict[str, object]]]:
        store = remote_store or StudioR2Store.from_settings()
        if store is None:
            raise HTTPException(status_code=503, detail="Studio R2 未配置")
        try:
            downloaded = store.download_images(selection.keys, temporary_dir)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=r2_error_detail(exc)) from exc
        return [item.path for item in downloaded], {
            str(item.provenance["sha256"]): item.provenance for item in downloaded
        }

    @app.post("/api/batches/r2")
    async def import_remote_batch(selection: RemoteSourceSelection) -> dict[str, object]:
        work_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix="studio-r2-import-", dir=work_root))
        try:
            paths, provenance = download_remote_sources(selection, temporary_dir)
            batch_dir, summary = await run_in_threadpool(
                create_batch,
                paths,
                work_root=work_root,
                holdout_ratio=selection.holdout_ratio,
                provenance_by_digest=provenance,
            )
            return {"batch": summary, "previews": roi_preview_paths(batch_dir)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @app.post("/api/batches")
    async def import_images(files: list[UploadFile] = File(...), holdout_ratio: float = Form(0.2)) -> dict[str, object]:
        work_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix="studio-import-", dir=work_root))
        try:
            paths = await _uploaded_paths(files, temporary_dir)
            batch_dir, summary = await run_in_threadpool(create_batch, paths, work_root=work_root, holdout_ratio=holdout_ratio)
            return {"batch": summary, "previews": roi_preview_paths(batch_dir)}
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @app.post("/api/batches/{batch_id}/sources")
    async def add_sources(batch_id: str, files: list[UploadFile] = File(...)) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        temporary_dir = Path(tempfile.mkdtemp(prefix="studio-append-", dir=work_root))
        try:
            paths = await _uploaded_paths(files, temporary_dir)
            return await run_in_threadpool(append_sources, batch_dir, paths)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @app.post("/api/batches/{batch_id}/remote-sources")
    async def add_remote_sources(batch_id: str, selection: RemoteSourceSelection) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        temporary_dir = Path(tempfile.mkdtemp(prefix="studio-r2-append-", dir=work_root))
        try:
            paths, provenance = download_remote_sources(selection, temporary_dir)
            return await run_in_threadpool(append_sources, batch_dir, paths, provenance_by_digest=provenance)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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

    @app.post("/api/batches/{batch_id}/dataset/export")
    async def export_finalized_dataset(batch_id: str) -> dict[str, object]:
        try:
            return await run_in_threadpool(export_dataset, _batch_dir(work_root, batch_id))
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
        resume_checkpoint = _resume_checkpoint(work_root, batch_dir, request.resume_checkpoint)
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
        return _list_resume_checkpoints(work_root, _batch_dir(work_root, batch_id))

    @app.post("/api/batches/{batch_id}/publication")
    def publish(batch_id: str, request: PublishStart) -> dict[str, object]:
        if not request.confirmed:
            raise HTTPException(status_code=422, detail="confirm publication before writing model artifacts to R2")
        batch_dir = _batch_dir(work_root, batch_id)
        checkpoint = _checkpoint_from_training_state(batch_dir)
        publication_root = batch_dir / "publication"
        latest_path = publication_root / "latest.json"
        if latest_path.is_file():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            if latest.get("status") == "publishing":
                raise HTTPException(status_code=409, detail="a model publication is already running")
        run_dir = publication_root / datetime.now(UTC).strftime("release-%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True)
        log_path = run_dir / "release.log"
        command = [str(ROOT / "training/release_rec_model.sh"), "--checkpoint", str(checkpoint)]
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state: dict[str, object] = {"pid": process.pid, "status": "publishing", "command": command, "log": str(log_path), "checkpoint": str(checkpoint)}
        payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        (run_dir / "publication.json").write_text(payload, encoding="utf-8")
        latest_path.write_text(payload, encoding="utf-8")
        return state

    @app.get("/api/batches/{batch_id}/publication")
    def publication_status(batch_id: str) -> dict[str, object]:
        batch_dir = _batch_dir(work_root, batch_id)
        state_path = batch_dir / "publication/latest.json"
        if not state_path.is_file():
            return {"status": "not_started", "log": "", "log_tail": ""}
        state: dict[str, object] = json.loads(state_path.read_text(encoding="utf-8"))
        changed = _poll_training_process(state, "publishing")
        log_path = Path(str(state.get("log", "")))
        state["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-48000:] if log_path.is_file() else ""
        if changed:
            persisted = {key: value for key, value in state.items() if key != "log_tail"}
            state_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return state

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

    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="studio")
    return app


def create_api_app() -> FastAPI:
    """Uvicorn reload factory for Studio development mode."""
    return create_app(frontend_dir=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only OCRKit Studio API and Vite-built frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--frontend-dir", type=Path, default=FRONTEND_DIST)
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn

    if args.reload:
        if not args.api_only or args.work_root != DEFAULT_WORK_ROOT:
            parser.error("--reload currently requires --api-only with the default work root")
        uvicorn.run(
            "training.studio.app:create_api_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(ROOT / "training/studio")],
        )
        return
    uvicorn.run(create_app(args.work_root, None if args.api_only else args.frontend_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
