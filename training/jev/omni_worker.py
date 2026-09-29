"""Persistent decision worker for the Jev-Omni MLX classifier.

Runs inside the model's own virtualenv (see ``training/README.md``); it must
stay dependency-free apart from the model checkout so the repository never
imports ``mlx``/``mlx-vlm`` itself. Protocol: one JSON task per stdin line,
one JSON response per stdout line. The first stdout line is a ready message
carrying the model identity used for decision provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    sys.path.insert(0, str(model_dir))
    from omni_mlx.classifier import Classifier  # noqa: E402  (resolved from the model checkout)

    started = time.perf_counter()
    classifier = Classifier(str(model_dir), str(args.calibration) if args.calibration else None)
    conversion = json.loads((model_dir / "conversion.json").read_text())
    ready = {
        "ready": True,
        "load_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": conversion.get("source"),
        "model_revision": conversion.get("revision"),
        "quantization_bits": conversion.get("bits"),
        "calibrated": bool(classifier.calibration),
    }
    sys.stdout.write(json.dumps(ready, ensure_ascii=False) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            result = classifier.predict(
                task["state"],
                task["question"],
                task["options"],
                image=task.get("image"),
                image_tokens=int(task.get("image_tokens", 70)),
            )
            response = {"task_id": task.get("task_id"), "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - per-task failure must not kill the worker
            try:
                task_id = json.loads(line).get("task_id")
            except json.JSONDecodeError:
                task_id = None
            response = {"task_id": task_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
