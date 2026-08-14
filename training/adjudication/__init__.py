"""Constrained text adjudication experiment (issue #4).

This package measures whether a constrained, text-only adjudicator can reduce
manual review for OCR terminology cases that ROI-scoped deterministic
normalization (#3) leaves unresolved, without increasing false confident
corrections.

The experiment is offline and replayable: it consumes reviewed-annotation
records that contain only text metadata (no images, no private object URLs, no
player identity, no submission state), and it stores captured provider output so
results can be reproduced without re-calling a provider.

Nothing here is a production OCRKit dependency. Provider adapters are optional,
fail-closed, and never become part of the main recognition, training, or model
release paths.
"""

from .adjudicator import (
    AdjudicationOutput,
    Adjudicator,
    HeuristicAdjudicator,
    OpenAICompatibleProvider,
    ProviderAdjudicator,
)
from .evaluate import DEFAULT_GATE, run_experiment
from .records import (
    EngineCandidate,
    NormalizationResult,
    ReviewedAnnotationRecord,
    canonicalize,
    compute_input_digest,
)

__all__ = [
    "DEFAULT_GATE",
    "AdjudicationOutput",
    "Adjudicator",
    "EngineCandidate",
    "HeuristicAdjudicator",
    "NormalizationResult",
    "OpenAICompatibleProvider",
    "ProviderAdjudicator",
    "ReviewedAnnotationRecord",
    "canonicalize",
    "compute_input_digest",
    "run_experiment",
]
