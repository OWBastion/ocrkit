"""Deterministic source-level train/holdout splitting.

Splits are decided by OCRKit, never by the platform snapshot. A stable hash of
``split_seed + snapshot_id + source_id`` maps each source screenshot to one
split, so a run is reproducible from the snapshot identity and the split rule
version, and every ROI/crop of one source stays in the same split (no
same-screenshot leakage between train and holdout).
"""

from __future__ import annotations

import hashlib

SPLIT_RULE_VERSION = "ocrkit-split-v1"

_TRAIN = "train"
_HOLDOUT = "holdout"


def source_split(
    source_id: str,
    snapshot_id: str,
    holdout_fraction: float,
    split_seed: str = "ocrkit-v1",
) -> str:
    digest = hashlib.sha256(f"{split_seed}:{snapshot_id}:{source_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return _HOLDOUT if bucket < holdout_fraction else _TRAIN


def split_sources(
    source_ids: list[str],
    snapshot_id: str,
    holdout_fraction: float,
    split_seed: str = "ocrkit-v1",
) -> dict[str, str]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1): {holdout_fraction}")
    return {source_id: source_split(source_id, snapshot_id, holdout_fraction, split_seed) for source_id in source_ids}


def split_rule_parameters(split_seed: str, holdout_fraction: float) -> dict[str, object]:
    return {
        "rule_version": SPLIT_RULE_VERSION,
        "split_seed": split_seed,
        "holdout_fraction": holdout_fraction,
    }
