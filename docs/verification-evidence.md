# Verification evidence

OCRKit's stricter specialization of the organization [testing](https://github.com/OWBastion/.github/blob/main/docs/testing-policy.md) and [verification](https://github.com/OWBastion/.github/blob/main/docs/verification-and-acceptance.md) policies. Read it before you add or change tests, fixtures, labels, expected values, or model evaluation.

Recognition expected values require an authority independent from the model or parser under test: reviewed labels, reproducible visible screenshot content, an accepted layout/API contract, or a real regression with provenance. Do not promote current model/parser output to fixture truth merely because it is stable, and do not rewrite labels or expectations only to make a new model pass. Aggregate accuracy, fixture counts, and test counts summarize evidence; they do not prove critical fields or uncertainty behavior correct. Current model versions, layout counts, fixture counts, and dataset cardinality are not durable correctness expectations unless that exact value is itself contractual.

Tests protect distinct recognition/API contracts and failure classes: valid evidence, unsupported/cropped/low-quality input, ambiguity, missing or conflicting fields, parser normalization, object/model failure, schema compatibility, and confidence/status behavior. Important changes cover both successful recognition and the relevant failure/uncertainty paths.

Material parser/layout/API/model-selection or confidence behavior changes should receive an independent attempt to falsify the implementation. Where practical, remove/invert the key parsing/validation behavior and confirm the targeted fixture or contract check fails again.

Do not add production API fields, debug endpoints, parser hooks, flags, state, or architecture layers solely to expose internals to tests.

Training evidence and decisive held-out evaluation evidence remain disjoint and distinguishable; the versioned source-level split documented in `training/README.md` enforces this, not convention. A sample present in both training data and the decisive evaluation set does not count as independent proof that a model change generalizes. Released model artifacts are versioned owner-side evidence; a mutable release pointer or channel is never fixture truth.
