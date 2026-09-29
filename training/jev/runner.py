"""Decision runners for the #25 Jev-Omni sidecar experiment.

A runner maps one :class:`PreReviewRecord` to a bounded ``DecisionOutput``.
``MockRunner`` exercises the pipeline without a model; ``OmniMlxRunner`` talks
to a persistent worker process running inside the model's own virtualenv, so
``mlx``/``mlx-vlm`` stay an optional, removable local dependency and never
enter OCRKit's runtime or training dependency graph. The runner is fail-closed:
transport errors, timeouts, and worker failures all produce an ``error``
output that routes the row to human review.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from .records import FIXED_OPTIONS, PreReviewRecord, compute_input_digest
from .tasks import build_task

WORKER_SCRIPT = Path(__file__).with_name("omni_worker.py")


class DecisionOutput(BaseModel):
    """One bounded decision with the provenance needed to reproduce it."""

    record_id: str
    input_digest: str
    runner: str
    status: Literal["ok", "error"]
    reason_code: str
    image_tokens: int
    options: list[str]
    selected_index: int | None = None
    selected_option: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    model: str | None = None
    model_revision: str | None = None
    latency_ms: float | None = None
    metrics: dict[str, float] | None = None


class Runner(Protocol):
    name: str

    def decide(self, record: PreReviewRecord, image_path: Path, image_tokens: int) -> DecisionOutput:
        ...

    def close(self) -> None:
        ...


def _error_output(
    record: PreReviewRecord,
    runner: str,
    image_tokens: int,
    reason_code: str,
    latency_ms: float | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        record_id=record.record_id,
        input_digest=compute_input_digest(record),
        runner=runner,
        status="error",
        reason_code=reason_code,
        image_tokens=image_tokens,
        options=list(record.options),
        latency_ms=latency_ms,
    )


class MockRunner:
    """Deterministic in-process runner for tests and pipeline dry runs.

    Picks the candidate option whose engine confidence is highest when it
    reaches ``accept_probability``; otherwise selects the none-correct option.
    """

    name = "mock"

    def __init__(self, accept_probability: float = 0.9) -> None:
        self.accept_probability = accept_probability
        self.calls = 0

    def decide(self, record: PreReviewRecord, image_path: Path, image_tokens: int) -> DecisionOutput:
        self.calls += 1
        best_index: int | None = None
        best_confidence = 0.0
        for engine in record.engines:
            confidence = engine.confidence or 0.0
            for index, option in enumerate(record.options):
                if option == engine.text and confidence > best_confidence:
                    best_index = index
                    best_confidence = confidence
        probabilities = [0.0] * len(record.options)
        if best_index is not None and best_confidence >= self.accept_probability:
            probabilities[best_index] = 0.97
        else:
            best_index = len(record.options) - len(FIXED_OPTIONS)
            probabilities[best_index] = 0.6
        remainder = (1.0 - probabilities[best_index]) / (len(record.options) - 1)
        probabilities = [remainder if index != best_index else probabilities[best_index] for index in range(len(record.options))]
        return DecisionOutput(
            record_id=record.record_id,
            input_digest=compute_input_digest(record),
            runner=self.name,
            status="ok",
            reason_code="ok",
            image_tokens=image_tokens,
            options=list(record.options),
            selected_index=best_index,
            selected_option=record.options[best_index],
            confidence=probabilities[best_index],
            probabilities=dict(zip(record.options, probabilities)),
            model="mock",
            latency_ms=0.0,
        )

    def close(self) -> None:
        return None


class OmniMlxRunner:
    """Persistent ``omni_mlx`` worker subprocess running in the model venv."""

    name = "omni-mlx"
    _MAX_RESPAWNS = 3

    def __init__(
        self,
        model_dir: Path,
        worker_python: Path,
        *,
        calibration: Path | None = None,
        timeout_seconds: float = 180.0,
        load_timeout_seconds: float = 600.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.worker_python = Path(worker_python)
        self.calibration = Path(calibration) if calibration else None
        self.timeout_seconds = timeout_seconds
        self.load_timeout_seconds = load_timeout_seconds
        self.model_identity: dict[str, str | None] = {}
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, object]] = queue.Queue()
        self._respawns = 0

    def _spawn(self) -> None:
        command = [str(self.worker_python), "-u", str(WORKER_SCRIPT), "--model-dir", str(self.model_dir)]
        if self.calibration:
            command += ["--calibration", str(self.calibration)]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
        assert self._process.stdout is not None
        self._responses = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self._process.stdout,), daemon=True).start()
        try:
            ready = self._responses.get(timeout=self.load_timeout_seconds)
        except queue.Empty as exc:
            self._kill()
            raise RuntimeError("omni worker did not become ready") from exc
        if not ready.get("ready"):
            self._kill()
            raise RuntimeError(f"omni worker failed to start: {ready.get('error')}")
        self.model_identity = {
            "model": str(ready.get("model")) if ready.get("model") else None,
            "model_revision": str(ready.get("model_revision")) if ready.get("model_revision") else None,
        }

    def _read_stdout(self, stream) -> None:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    self._responses.put(json.loads(line))
                except json.JSONDecodeError:
                    self._responses.put({"ok": False, "error": f"worker emitted invalid JSON: {line[:200]}"})

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None or self._respawns:
            self._respawns += 1
        if self._respawns > self._MAX_RESPAWNS:
            raise RuntimeError("omni worker exhausted respawn attempts")
        self._spawn()

    def _kill(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process = None

    def decide(self, record: PreReviewRecord, image_path: Path, image_tokens: int) -> DecisionOutput:
        digest = compute_input_digest(record)
        try:
            self._ensure_worker()
        except RuntimeError:
            return _error_output(record, self.name, image_tokens, "worker_unavailable")
        assert self._process is not None and self._process.stdin is not None
        task = build_task(record, str(image_path)) | {"image_tokens": image_tokens}
        started = time.perf_counter()
        try:
            self._process.stdin.write(json.dumps(task, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            response = self._responses.get(timeout=self.timeout_seconds)
        except (queue.Empty, BrokenPipeError, OSError):
            self._kill()
            return _error_output(
                record, self.name, image_tokens, "timeout", round((time.perf_counter() - started) * 1000, 1)
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if not response.get("ok"):
            return _error_output(record, self.name, image_tokens, "worker_failure", latency_ms)
        result = response.get("result")
        if not isinstance(result, dict) or result.get("prediction_index") is None:
            return _error_output(record, self.name, image_tokens, "invalid_output", latency_ms)
        probabilities = result.get("probabilities")
        selected_index = int(result["prediction_index"])
        if not 0 <= selected_index < len(record.options):
            return _error_output(record, self.name, image_tokens, "invalid_output", latency_ms)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
        confidence = None
        if isinstance(probabilities, dict):
            confidence = probabilities.get(record.options[selected_index])
        return DecisionOutput(
            record_id=record.record_id,
            input_digest=digest,
            runner=self.name,
            status="ok",
            reason_code="ok",
            image_tokens=image_tokens,
            options=list(record.options),
            selected_index=selected_index,
            selected_option=record.options[selected_index],
            confidence=round(float(confidence), 6) if isinstance(confidence, (int, float)) else None,
            probabilities={str(key): float(value) for key, value in probabilities.items()} if isinstance(probabilities, dict) else None,
            model=self.model_identity.get("model"),
            model_revision=self.model_identity.get("model_revision"),
            latency_ms=metrics.get("elapsed_ms") if isinstance(metrics, dict) and isinstance(metrics.get("elapsed_ms"), (int, float)) else latency_ms,
            metrics={str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))} if metrics else None,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()


def build_runner(
    kind: str,
    *,
    model_dir: Path | None = None,
    worker_python: Path | None = None,
    calibration: Path | None = None,
    timeout_seconds: float = 180.0,
) -> Runner:
    if kind == "mock":
        return MockRunner()
    if kind == "omni-mlx":
        if model_dir is None or worker_python is None:
            raise ValueError("omni-mlx runner requires --model-dir and --worker-python")
        return OmniMlxRunner(model_dir, worker_python, calibration=calibration, timeout_seconds=timeout_seconds)
    raise ValueError(f"unknown runner kind: {kind!r}")
