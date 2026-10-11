"""Score a candidate artifact against the current stable model under identical conditions.

Fixed accuracy floors go stale whenever the layout, the parsers or the fixtures change, so the question asked here
is relative: measured now, on the same code and the same inputs, is the candidate worse than the model in service?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Importing the app resolves a model from settings; the artifact under test must be the only model in play.
os.environ["OCRKIT_MODEL_MANIFEST_KEY"] = ""
os.environ["OCRKIT_MODEL_RELEASE_CHANNEL_KEY"] = ""
# The service's default cache is a production path; the model in service is cached under the training workspace.
os.environ.setdefault("OCRKIT_MODEL_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "training/.work/cache/model-in-service"))

from scripts import batch_eval  # noqa: E402
from training.scripts import evaluate_rec_holdout  # noqa: E402

FIXTURE_CASES = Path("datasets/fixtures/challenge/cases.json")
FIXTURE_IMAGES = Path("datasets/fixtures/challenge")
COMPATIBILITY_CONFIG = Path("configs/bastion_screenshot_compatibility.json")


def measure(model_config: Path, holdout: tuple[Path, Path] | None) -> dict[str, Any]:
    floor = json.loads(COMPATIBILITY_CONFIG.read_text(encoding="utf-8"))["producer_contract"]["minimum_released_version"]
    fixture = batch_eval.evaluate(FIXTURE_CASES, FIXTURE_IMAGES, model_config, minimum_version=floor)
    fixture["run_code"] = batch_eval.evaluate(batch_eval.DEFAULT_RUN_CODE_CASES, batch_eval.DEFAULT_RUN_CODE_IMAGES_DIR, model_config)
    return {
        "fixture": fixture,
        "holdout": evaluate_rec_holdout.evaluate(holdout[0], holdout[1], model_config) if holdout else None,
    }


def decide(candidate: dict[str, Any], stable: dict[str, Any] | None) -> tuple[str, list[str]]:
    """Return ``passed``, ``failed`` or ``no_baseline`` with the reasons for any failure."""
    reasons: list[str] = []
    run_code = candidate["fixture"]["run_code"]["field_accuracy"]
    if run_code < 1.0:
        reasons.append(f"run-code fixtures must stay exact (candidate {run_code:.4f})")
    if stable is None:
        return ("failed" if reasons else "no_baseline"), reasons

    metrics = candidate["fixture"].get("field_metrics", {})
    for name, base in sorted(stable["fixture"].get("field_metrics", {}).items()):
        now = metrics.get(name, {"matched": 0, "total": 0})
        if now["matched"] < base["matched"]:
            reasons.append(f"regresses {name}: {now['matched']}/{now['total']} vs {base['matched']}/{base['total']} on fixtures")
    if candidate["fixture"]["field_accuracy"] < stable["fixture"]["field_accuracy"]:
        reasons.append(
            f"fixture accuracy {candidate['fixture']['field_accuracy']:.4f} is below the model in service {stable['fixture']['field_accuracy']:.4f}"
        )
    if candidate["holdout"] and stable["holdout"] and candidate["holdout"]["accuracy"] < stable["holdout"]["accuracy"]:
        reasons.append(
            f"holdout recognition {candidate['holdout']['accuracy']:.4f} is below the model in service {stable['holdout']['accuracy']:.4f}"
        )
    return ("failed" if reasons else "passed"), reasons


def _summary(measured: dict[str, Any]) -> dict[str, Any]:
    fixture = measured["fixture"]
    holdout = measured["holdout"]
    return {
        "fixture": {
            "field_accuracy": fixture["field_accuracy"],
            "matched_fields": fixture["matched_fields"],
            "total_fields": fixture["total_fields"],
            "run_code_accuracy": fixture["run_code"]["field_accuracy"],
            "field_metrics": {name: {"matched": item["matched"], "total": item["total"]} for name, item in fixture.get("field_metrics", {}).items()},
        },
        "holdout": None if holdout is None else {"accuracy": holdout["accuracy"], "matched": holdout["matched"], "total": holdout["total"], "by_roi": holdout["by_roi"]},
    }


def load_stable() -> tuple[str, str, Path] | None:
    """The model in service, or None when there is no stable channel to compare with."""
    from app.core.config import settings

    if not (settings.r2_endpoint_url and settings.r2_access_key_id and settings.r2_secret_access_key and settings.r2_default_bucket):
        print("no baseline: model storage is not configured", file=sys.stderr)
        return None
    from app.main import _load_model, _model_store
    from app.model_artifacts import load_release_channel
    from app.model_artifacts.release import STABLE_CHANNEL_KEY
    from app.storage.r2_client import ObjectNotFoundError

    try:
        manifest_key = load_release_channel(_model_store(), settings.r2_default_bucket, STABLE_CHANNEL_KEY).manifest_key
    except ObjectNotFoundError:
        print("no baseline: there is no stable channel yet", file=sys.stderr)
        return None
    version, config = _load_model(manifest_key)
    return version, manifest_key, Path(str(config))


def compare(artifact_dir: Path, holdout: tuple[Path, Path] | None, *, use_stable: bool = True) -> dict[str, Any]:
    stable_model = load_stable() if use_stable else None  # resolve first: fail before the slow measurements
    candidate = measure(artifact_dir / "rapidocr.yaml", holdout)
    stable = measure(stable_model[2], holdout) if stable_model else None
    verdict, reasons = decide(candidate, stable)
    status = "failed" if verdict == "failed" else "passed"

    fixture_report = {**candidate["fixture"], "status": status}
    (artifact_dir / "fixture_report.json").write_text(json.dumps(fixture_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if candidate["holdout"] is not None:
        holdout_report = {**candidate["holdout"], "status": status}
        (artifact_dir / "holdout_report.json").write_text(json.dumps(holdout_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "verdict": verdict,
        "reasons": reasons,
        "stable": {"version": stable_model[0], "manifest_key": stable_model[1]} if stable_model else None,
        "candidate": _summary(candidate),
        "baseline": _summary(stable) if stable else None,
    }
    (artifact_dir / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a candidate artifact with the model in service.")
    parser.add_argument("--candidate-artifact", required=True, type=Path)
    parser.add_argument("--holdout-labels", type=Path)
    parser.add_argument("--holdout-images-root", type=Path)
    parser.add_argument("--no-stable", action="store_true", help="skip the baseline (offline experiments only)")
    args = parser.parse_args()
    if bool(args.holdout_labels) != bool(args.holdout_images_root):
        parser.error("--holdout-labels and --holdout-images-root go together")
    holdout = (args.holdout_labels, args.holdout_images_root) if args.holdout_labels else None
    report = compare(args.candidate_artifact, holdout, use_stable=not args.no_stable)
    candidate, baseline = report["candidate"], report["baseline"]
    print(f"fixtures: candidate {candidate['fixture']['field_accuracy']:.4f}" + (f" | in service {baseline['fixture']['field_accuracy']:.4f}" if baseline else ""))
    if candidate["holdout"]:
        print(f"holdout:  candidate {candidate['holdout']['accuracy']:.4f}" + (f" | in service {baseline['holdout']['accuracy']:.4f}" if baseline and baseline["holdout"] else ""))
    print(f"verdict: {report['verdict']}" + (f" against {report['stable']['version']}" if report["stable"] else " (no baseline)"))
    for reason in report["reasons"]:
        print(f"  - {reason}", file=sys.stderr)
    if report["verdict"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
