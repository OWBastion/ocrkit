# OCRKit Agent Guide

OCRKit is the Bastion ecosystem's stateless screenshot-recognition service and OCR model-lifecycle owner. [OWBastion organization policy](https://github.com/OWBastion/.github/blob/main/README.md) owns shared engineering, testing, verification, entropy, and delivery policy. This file specializes OCRKit responsibility, recognition/evidence invariants, risk routing, privacy, and local validation. Keep mutable model/layout/version inventories in their live sources rather than here.

## Repository role

OCRKit extracts structured evidence from known Bastion screenshot layouts. It owns image validation/normalization, ROI extraction and preprocessing, OCR invocation, field parsing/normalization, field confidence/warnings, recognition API behavior, and OCR model artifact/training/evaluation workflows.

`OWBastion/owbastion.com` owns screenshot submission state, identity/business rules, challenge matching, review, corrections/adoption, and final grants. `OWBastion/Bastion` owns game behavior and the game-side HUD/content that produces screenshot evidence. `OWBastion/qqbot` owns QQ channel behavior.

OCRKit must not decide whether a player deserves a title, whether a submission should be approved, or whether OCR evidence should cause a grant. Add extracted facts as evidence, not business conclusions.

## Start here

For substantive recognition/model work:

1. Read the linked Issue and `README.md`, then inspect the smallest relevant source, configuration, tests/fixtures, and specialist documentation.
2. Resolve current API shapes, supported layouts, model versions/channels, thresholds, and deployment details from their authoritative source; this guide is not a mutable inventory.
3. Compare the Issue contract, current recognition/API contract, and implementation reality. Surface material mismatches instead of inventing a new business rule, layout contract, public contract, or cross-service ownership decision.
4. Verify recognition behavior against evidence independent from the implementation, including ambiguous/unsupported/low-quality paths where relevant.

## Repository delivery constraints

Follow [organization PR delivery policy](https://github.com/OWBastion/.github/blob/main/docs/pr-delivery.md). Model publication/channel changes, R2 writes, and production recognition checks remain subject to local owner authorization.

## Risk routing

- Current API routes, response fields, configuration, runtime behavior: `README.md`, API/schema source, and contract tests.
- Screenshot layouts, ROIs, normalization, preprocessing, quality detection: current `configs/`, image/layout source, manifests, and representative fixtures.
- Field parsing, aliases, confidence/status/warning behavior: parser/recognition source plus focused fixtures/tests.
- Model artifacts, training, evaluation, publication, rollback: `training/README.md`, model scripts/source, manifests, and release workflows.
- Rust image-preflight tooling: `rust/README.md`, Rust source/tests, and layout-manifest generation checks.
- R2/object access, private screenshots, debug crops, credentials, retention: current storage/config source and deployment/runtime docs; privacy is a blocking correctness requirement.
- Platform/Bastion contract changes: inspect the owning repository contract and integrate OCRKit separately rather than duplicating authority here.

## Recognition invariants

- OCRKit serves known Bastion screenshot layouts, not generic full-screen OCR or free-form visual understanding.
- Prefer deterministic image/layout handling, ROI extraction, parsing, and validation before expanding model complexity when they can solve the measured failure.
- Train/fine-tune models only when measured evidence shows deterministic preprocessing/parsing is insufficient for the target field/layout.
- Production inference must not depend on Apple-only APIs. Keep training-only dependencies out of production runtime; do not make a large multimodal model the primary path without an explicit architecture decision.
- Low-confidence, incomplete, ambiguous, or explicitly unsupported output is preferable to a confidently fabricated value.
- Recognition output exposes evidence quality/confidence/status sufficiently for the platform to make its own business decision; OCRKit must not encode approval/grant conclusions.
- Do not hard-code one screenshot's expected values into production parsers or recognition logic.
- New/changed layout or field support needs representative regression evidence and preserves supported behavior unless deprecation/change is explicitly approved.
- Breaking recognition-response changes require an explicit compatible/versioned migration plan with affected consumers; do not silently repurpose fields.
- Released model artifacts are immutable/versioned and verifiable before use; changing a mutable channel/pointer must not rewrite released artifacts.
- Recognition requests should be retry-safe; external/object-store interactions need bounded image-size, timeout, concurrency, and memory behavior.
- The service remains stateless with respect to platform business data; a verified local model cache is not business truth.

## Privacy and data safety

Treat player screenshots, OCR debug crops, private object keys/URLs, credentials, and production evidence as private data.

Never commit production screenshots or copied private payloads. Do not log image bytes, signed/private URLs, credentials, or unrelated personal data. Promoting production/reviewer evidence into fixtures or training data requires explicit approval, provenance, and appropriate privacy/retention handling.

Object-mode recognition must restrict reads to allowed storage boundaries and reject traversal/unexpected locations. Browser/client-facing surfaces must not receive storage credentials.

## Recognition verification

Apply the [organization testing policy](https://github.com/OWBastion/.github/blob/main/docs/testing-policy.md) and [verification policy](https://github.com/OWBastion/.github/blob/main/docs/verification-and-acceptance.md). Recognition expectations require independent evidence such as reviewed fixture truth, reproducible visible screenshot evidence, an accepted API/layout contract, or a real regression with provenance. Do not change an expected value merely because the new implementation emits it.

Important changes cover success and relevant failure/uncertainty paths: unsupported/cropped/low-quality inputs, missing/conflicting fields, parsing ambiguity, object/model failures, and schema compatibility as applicable.

Material parser/layout/API/model-selection/confidence changes should receive an independent falsification pass. State the claim; where practical remove/invert the key parsing or validation behavior and confirm the targeted fixture/contract check fails again.

Do not add OCR response fields, debug routes, parser hooks, or architecture layers solely to expose internal recognition behavior to tests.

## Local validation

Use current repository docs/scripts as the command source of truth. Run focused recognition/parser/layout tests first, then broader Python test/lint/type/build gates required by the change. When Rust preflight/layout manifests are touched, run the documented Rust/layout checks. When training/model publication behavior changes, follow `training/README.md` and the relevant evaluation/release workflow rather than inferring a process from this file.
