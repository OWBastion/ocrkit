# OCRKit Agent Guide

OCRKit is the Bastion ecosystem's stateless screenshot-recognition service and OCR model-lifecycle owner. [OWBastion organization routing](https://github.com/OWBastion/.github/blob/main/AGENTS.md) owns repository ownership, shared policy routing, and global invariants; this file specializes OCRKit responsibility, recognition/evidence invariants, risk routing, privacy, and local validation.

## Repository role

OCRKit extracts structured evidence from known Bastion screenshot layouts. It owns image validation/normalization, ROI extraction and preprocessing, OCR invocation, field parsing/normalization, field confidence/warnings, recognition API behavior, and OCR model artifact/training/evaluation workflows.

`OWBastion/owbastion.com` owns screenshot submission state, identity/business rules, challenge matching, review, corrections/adoption, and final grants. `OWBastion/Bastion` owns game behavior and the game-side HUD/content that produces screenshot evidence. `OWBastion/qqbot` owns QQ channel behavior.

OCRKit must not decide whether a player deserves a title, whether a submission should be approved, or whether OCR evidence should cause a grant. Add extracted facts as evidence, not business conclusions.

## Contribution to the organization goal

Serves the [organization product goal](https://github.com/OWBastion/.github/blob/main/docs/product-goal.md) by keeping score verification light for players: recognize known Bastion screenshots well enough that ordinary players rarely need more than a screenshot, and surface uncertainty so human review stays targeted. It does not grow into general OCR or visual understanding.

## Start here

For substantive work:

1. Read the linked Issue and `README.md`, then inspect the smallest relevant source, configuration, tests/fixtures, and specialist documentation.
2. Resolve current API shapes, supported layouts, model versions/channels, thresholds, and deployment details from their current authoritative source; do not use this root guide as a mutable inventory.
3. Compare the Issue contract, current recognition/API contract, and implementation reality. Report material mismatches instead of inventing a new business rule, layout contract, or cross-service behavior.
4. Verify recognition behavior against evidence independent from the implementation being changed, including ambiguous/unsupported/low-quality paths where relevant.

## Risk routing

- Current API routes, response fields, configuration, and runtime behavior: `README.md`, API/schema source, and contract tests.
- Screenshot layouts, ROIs, normalization, preprocessing, and quality detection: current `configs/`, image/layout source, manifests, and representative fixtures.
- Field parsing, aliases, confidence/status/warning behavior: parser/recognition source plus focused fixtures/tests.
- Model artifacts, training, evaluation, publication, or rollback: `training/README.md`, model-related scripts/source, manifests, and release workflows.
- Rust image-preflight tooling: `rust/README.md`, Rust source/tests, and layout-manifest generation checks.
- R2/object access, private screenshots, debug crops, credentials, or retention: current storage/config source and deployment/runtime documentation; privacy is a blocking correctness requirement.
- Platform/Bastion contract changes: inspect the owning repository contract as needed and integrate OCRKit separately rather than duplicating authority here.

## Recognition invariants

- OCRKit is for known Bastion screenshot layouts, not a generic full-screen OCR or free-form visual-understanding product.
- Prefer deterministic image/layout handling, ROI extraction, parsing, and validation before introducing or expanding model complexity when they can solve the measured failure.
- Train or fine-tune models only when measured evidence shows deterministic preprocessing/parsing is insufficient for the target field/layout; fallback models, routing layers, or multimodal paths need the same measured recognition gap, not speculative improvement.
- Production inference must not depend on Apple-only APIs. Keep training-only dependencies out of the production runtime, and do not make a large multimodal model the primary recognition path without an explicit architecture decision.
- A low-confidence, incomplete, ambiguous, or explicitly unsupported result is preferable to a confidently fabricated value.
- Recognition output must expose evidence quality/confidence/status sufficiently for the platform to make its own business decision; OCRKit must not encode approval/grant conclusions.
- Do not hard-code a particular screenshot's expected values into production parsers or recognition logic.
- New or changed layout/field support needs representative regression evidence and must preserve supported behavior unless a deprecation/change is explicitly approved.
- Breaking recognition-response changes require an explicit versioned/compatible migration plan with affected consumers; do not silently repurpose existing fields.
- Released model artifacts are immutable/versioned and must be verifiable before use; changing a mutable release pointer/channel must not rewrite previously released artifacts.
- Recognition requests should be retry-safe and external/object-store interactions must have bounded resource behavior appropriate to image size, timeout, concurrency, and memory risk.
- The service remains stateless with respect to platform business data; a verified local model cache is not a business source of truth.

## Privacy and data safety

Treat player screenshots, OCR debug crops, private object keys/URLs, credentials, and production evidence as private data.

Never commit production screenshots or copied private payloads to the public repository. Do not log image bytes, signed/private URLs, credentials, or unrelated personal data. Training or fixture promotion from production/reviewer evidence requires explicit approval, provenance, and the appropriate privacy/retention handling; production evidence is not automatically training data.

Object-mode recognition must restrict reads to explicitly allowed storage boundaries and reject traversal/unexpected locations. Browser/client-facing surfaces must not receive storage credentials.

## Verification

Apply the organization [testing](https://github.com/OWBastion/.github/blob/main/docs/testing-policy.md), [verification](https://github.com/OWBastion/.github/blob/main/docs/verification-and-acceptance.md), and [engineering quality](https://github.com/OWBastion/.github/blob/main/docs/engineering-quality.md) policies; the rules below are OCRKit's stricter evidence specialization.

Recognition expected values require an authority independent from the model or parser under test: reviewed labels, reproducible visible screenshot content, an accepted layout/API contract, or a real regression with provenance. Do not promote current model/parser output to fixture truth merely because it is stable, and do not rewrite labels or expectations only to make a new model pass. Aggregate accuracy, fixture counts, and test counts summarize evidence; they do not prove critical fields or uncertainty behavior correct. Current model versions, layout counts, fixture counts, and dataset cardinality are not durable correctness expectations unless that exact value is itself contractual.

Tests protect distinct recognition/API contracts and failure classes: valid evidence, unsupported/cropped/low-quality input, ambiguity, missing or conflicting fields, parser normalization, object/model failure, schema compatibility, and confidence/status behavior. Important changes cover both successful recognition and the relevant failure/uncertainty paths.

Material parser/layout/API/model-selection or confidence behavior changes should receive an independent attempt to falsify the implementation. Where practical, remove/invert the key parsing/validation behavior and confirm the targeted fixture or contract check fails again.

Do not add production API fields, debug endpoints, parser hooks, flags, state, or architecture layers solely to expose internals to tests.

Training evidence and decisive held-out evaluation evidence remain disjoint and distinguishable; the versioned source-level split documented in `training/README.md` enforces this, not convention. A sample present in both training data and the decisive evaluation set does not count as independent proof that a model change generalizes. Released model artifacts are versioned owner-side evidence; a mutable release pointer or channel is never fixture truth.

## Entropy

Apply the organization [entropy policy](https://github.com/OWBastion/.github/blob/main/docs/entropy-policy.md). OCRKit cleanup targets duplicate preprocess paths, obsolete model fallbacks, stale layout compatibility, duplicated parser normalization, generated artifacts with no surviving owner, and temporary migration adapters. Map platform, API, and model consumers before removing an apparently redundant recognition path; do not remove uncertainty/status reporting, validation, privacy boundaries, artifact verification, or compatibility behavior while consumers still rely on it.

## Local validation

Use current repository documentation and scripts as the command source of truth. Run focused recognition/parser/layout tests first, then the broader Python test/lint/type/build gates required by the change. When Rust preflight code/layout manifests are touched, run the documented Rust/layout checks. When training/model publication behavior is touched, follow `training/README.md` and the relevant release/evaluation workflow rather than inferring a process from this file.

Deployment, model publication, R2 writes, production recognition checks, and other external writes are separate from local validation and require the appropriate explicit authorization/configuration.
