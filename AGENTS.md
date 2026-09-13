# OCRKit Agent Guide

OCRKit is the Bastion ecosystem's stateless screenshot-recognition service and OCR model-lifecycle owner. Workspace guidance owns shared engineering policy; this file specializes OCRKit responsibility, recognition/evidence invariants, risk routing, privacy, and local validation.

## Repository role

OCRKit extracts structured evidence from known Bastion screenshot layouts. It owns image validation/normalization, ROI extraction and preprocessing, OCR invocation, field parsing/normalization, field confidence/warnings, recognition API behavior, and OCR model artifact/training/evaluation workflows.

`OWBastion/owbastion.com` owns screenshot submission state, identity/business rules, challenge matching, review, corrections/adoption, and final grants. `OWBastion/Bastion` owns game behavior and the game-side HUD/content that produces screenshot evidence. `OWBastion/qqbot` owns QQ channel behavior.

OCRKit must not decide whether a player deserves a title, whether a submission should be approved, or whether OCR evidence should cause a grant. Add extracted facts as evidence, not business conclusions.

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

- Prefer deterministic image/layout handling, ROI extraction, parsing, and validation before introducing or expanding model complexity when they can solve the measured failure.
- A low-confidence, incomplete, ambiguous, or explicitly unsupported result is preferable to a confidently fabricated value.
- Recognition output must expose evidence quality/confidence/status sufficiently for the platform to make its own business decision; OCRKit must not encode approval/grant conclusions.
- Do not hard-code a particular screenshot's expected values into production parsers or recognition logic.
- New or changed layout/field support needs representative regression evidence and must preserve supported behavior unless a deprecation/change is explicitly approved.
- Production inference and offline training concerns remain separable. Training-only dependencies or data must not leak into the production path without an explicit architecture decision.
- Released model artifacts are immutable/versioned and must be verifiable before use; changing a mutable release pointer/channel must not rewrite previously released artifacts.
- Recognition requests and external/object-store interactions must have bounded resource behavior appropriate to image size, timeout, concurrency, and memory risk.
- The service remains stateless with respect to platform business data; a verified local model cache is not a business source of truth.

## Privacy and data safety

Treat player screenshots, OCR debug crops, private object keys/URLs, credentials, and production evidence as private data.

Never commit production screenshots or copied private payloads to the public repository. Do not log image bytes, signed/private URLs, credentials, or unrelated personal data. Training or fixture promotion from production/reviewer evidence requires explicit approval, provenance, and the appropriate privacy/retention handling; production evidence is not automatically training data.

Object-mode recognition must restrict reads to explicitly allowed storage boundaries and reject traversal/unexpected locations. Browser/client-facing surfaces must not receive storage credentials.

## Verification

Recognition expectations require an independent basis such as reviewed fixture truth, reproducible visible screenshot evidence, an accepted API/layout contract, or a real regression with provenance. Do not change an expected value merely because the new implementation emits it.

Important changes should cover both successful recognition and relevant failure/uncertainty paths: unsupported/cropped/low-quality inputs, missing or conflicting fields, parsing ambiguity, object/model failures, and schema compatibility as applicable.

Material parser/layout/API/model-selection or confidence behavior changes should receive an independent attempt to falsify the implementation. Where practical, remove/invert the key parsing/validation behavior and confirm the targeted fixture or contract check fails again.

Do not add test-only production APIs, hooks, or architecture layers solely to make internal behavior observable.

## Local validation

Use current repository documentation and scripts as the command source of truth. Run focused recognition/parser/layout tests first, then the broader Python test/lint/type/build gates required by the change. When Rust preflight code/layout manifests are touched, run the documented Rust/layout checks. When training/model publication behavior is touched, follow `training/README.md` and the relevant release/evaluation workflow rather than inferring a process from this file.

Deployment, model publication, R2 writes, production recognition checks, and other external writes are separate from local validation and require the appropriate explicit authorization/configuration.
