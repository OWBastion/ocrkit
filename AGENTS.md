# AGENTS.md

> Ecosystem contract version: `1.0`
> Repository: `OWBastion/ocrkit`
> Role: stateless screenshot-recognition service and OCR model lifecycle

## 1. Mission

OCRKit extracts structured fields from known Bastion Escape 3 screenshot layouts.

It is not a generic OCR product and not an achievement-review engine. Its output is evidence used by `owbastion.com`, which owns business rules and final decisions.

The service should identify fields such as:

- challenge completion state;
- hero progress;
- player name;
- deaths and skips;
- elapsed time;
- map and difficulty;
- game version;
- supported mode or layout identifiers when visible.

The goal is to extract specific HUD fields from known screenshot layouts, including:

- top-left challenge progress and statistics;
- center completion banner;
- top-right map, difficulty, and version information;
- optional bottom-left hero/status information.

The system should return structured JSON suitable for a Web API, automation pipeline, or leaderboard service. OCRKit provides evidence; it does not make leaderboard, approval, or title decisions.

## 2. Ecosystem Position

| Repository | Ownership |
| --- | --- |
| `OWBastion/Bastion` | Released game definitions and screenshot HUD contract |
| `OWBastion/owbastion.com` | OCR orchestration, review, corrections, training feedback |
| `OWBastion/qqbot` | QQ user entry and notifications |
| `OWBastion/ocrkit` | Recognition, parsing, confidence, model publication |

OCRKit must not know whether a player deserves a title.

## 3. Hard Responsibility Boundary

OCRKit owns:

- image decoding and validation;
- layout normalization and ROI extraction;
- image preprocessing;
- OCR engine invocation;
- tolerant field parsing and normalization;
- field-level confidence and warnings;
- model artifact loading, verification, and rollback;
- offline dataset preparation, training, evaluation, and release gates;
- stable recognition API contracts.

OCRKit does not own:

- QQ identity or player-account binding;
- screenshot-submission state;
- challenge definitions or title rules;
- automatic approval policy;
- administrator review UI;
- title grants or GitHub PR creation;
- public leaderboards or player progression.

## 4. Recognition Principles

Prefer deterministic image processing, ROI cropping, field parsing, and validation over model training.

Train or fine-tune models only after measured evidence shows the existing OCR, preprocessing, and parsing pipeline is insufficient.

1. Keep production inference independent of Apple-only APIs.
2. Keep PaddleOCR training dependencies out of the production image.
3. Do not use large multimodal models in the primary production path without an explicit architecture decision.
4. Never hardcode expected fixture values into parsers.
5. A low-confidence or incomplete result is preferable to a confidently wrong fabricated value.

The expected processing pipeline is:

```text
image upload
→ normalize image size
→ crop configured ROIs
→ preprocess each ROI
→ OCR each ROI
→ parse fields with tolerant rules
→ normalize and validate fields
→ return structured JSON
```

## 5. Non-Goals

- Do not build a generic OCR system.
- Do not OCR the entire screenshot and then infer fields from all detected text.
- Do not use Core ML.
- Do not design the system around Apple-only deployment.
- Do not hardcode a single screenshot's values into the parser.
- Do not use large multimodal LLMs for the production recognition path unless explicitly requested.
- Do not introduce custom model training without measuring the deterministic pipeline first.

## 6. API Contract

Every successful response should be traceable. The target response envelope should include:

```json
{
  "schema_version": "1",
  "request_id": "...",
  "engine": "rapidocr",
  "model_version": "...",
  "layout_version": "...",
  "ok": true,
  "data": {},
  "fields": {},
  "warnings": [],
  "quality": {}
}
```

Each critical field should expose, directly or through the `fields` object:

- parsed value;
- confidence;
- source ROI;
- normalization or alias applied;
- missing, ambiguous, or conflict state where relevant.

Do not make business decisions such as `eligible_for_title` or `approve_submission` part of the OCR response.

Breaking response changes require a new schema version and coordinated consumer migration.

## 7. Object Storage Contract

Production object-mode recognition may read only explicitly allowed private R2 objects.

Maintain namespace separation:

```text
uploads/                         user screenshot evidence
models/pp-ocrv6-small/<version>/ versioned model artifacts
```

Rules:

- reject traversal and unexpected prefixes;
- use bucket allow-lists;
- enforce image size, MIME, decode, and timeout limits;
- never expose R2 credentials or signed object URLs in responses or logs;
- model artifacts are immutable and content-addressed by manifest hashes;
- user screenshots must not be bundled into service images or model releases.

## 8. Layout and Image Quality

Do not blindly convert every image into a valid result. Recognition should detect and report:

- unsupported aspect ratio;
- likely crop or missing HUD regions;
- image too small or excessively compressed;
- unexpected layout version;
- conflicting completion indicators;
- fields outside plausible ranges;
- missing version, map, difficulty, or player identity.

New layout support must be versioned and regression-tested. Preserve old supported layouts unless a documented deprecation is approved.

## 9. Model Lifecycle

Model releases must be immutable and versioned.

Required release flow:

```text
reviewed labels
→ training or fine-tuning
→ isolated holdout evaluation
→ end-to-end fixture evaluation
→ minimum field-accuracy gate
→ full test suite
→ export inference artifacts
→ build manifest with hashes
→ upload under a new version prefix
→ download and checksum verification
→ RapidOCR load verification
→ controlled deployment
```

Never overwrite a released model prefix. Rollback must require only selecting an earlier manifest.

Training data, fixtures, and production evidence have different purposes:

- training set: reviewed examples used for optimization;
- holdout set: isolated examples not used for training decisions;
- fixture regression set: stable service-level cases;
- production evidence: private user data, not automatically a training sample.

A reviewer correction from `owbastion.com` becomes training data only after explicit approval and provenance recording.

## 10. Evaluation Standard

Do not optimize only aggregate character accuracy. Track at minimum:

- exact match per field;
- all-critical-fields-correct rate;
- false-positive completion rate;
- map and difficulty accuracy;
- player-name accuracy;
- numeric-field accuracy;
- unsupported-layout rejection quality;
- latency percentiles;
- memory usage;
- error rate by screenshot resolution and source.

For an approval workflow, false confident positives are higher risk than missing values. Release gates should reflect this asymmetry.

## 11. Privacy and Retention

- Treat all player screenshots and OCR debug crops as private data.
- Do not commit production screenshots to the public repository.
- Do not retain user images inside application logs.
- Debug responses must be restricted to trusted service callers or non-production environments.
- Training exports must remove unrelated metadata and preserve submission provenance separately.
- Respect deletion or retention policies defined by the platform.

## 12. Reliability and Operations

- Recognition requests must be retry-safe.
- The service should remain stateless except for a verified local model cache.
- Health output must include engine, application version, and loaded model version.
- Startup must fail when an explicitly configured model manifest is missing, incomplete, or checksum-invalid.
- Network and object-store timeouts must be bounded.
- Concurrency must be limited to prevent memory exhaustion.
- Logs must include request/correlation IDs but not image bytes or private URLs.

## 13. Code Organization

Keep explicit modules for:

```text
app/api/               HTTP schemas and routing
app/image/             decoding, normalization, ROI, preprocessing
app/ocr/               engine adapters
app/parser/            field-specific parsers
app/model_artifacts/   manifest and cache validation
app/storage/           object-store adapters
training/              offline-only model workflows
tests/                 unit, contract, fixture, and integration tests
```

Write simple, explicit Python:

- prefer small pure functions for parsers and normalizers;
- avoid hidden global state;
- avoid hardcoded ROI coordinates in parser code;
- avoid mixing OCR logic, image preprocessing, and field parsing in one function;
- use type hints for public functions;
- use Pydantic models for API input and output schemas.

## 14. Cross-Repository Change Rules

When Bastion changes HUD layout, labels, map aliases, or version formatting:

- update layout/parser configuration;
- add representative fixtures;
- preserve compatibility where possible;
- coordinate the new output schema with `owbastion.com`;
- document the minimum compatible game version.

When the platform requests a new extracted field:

- confirm it is visibly and reliably present;
- add it as OCR evidence, not a business conclusion;
- version contracts when required;
- define confidence and warning behavior;
- add tests before production use.

## 15. Definition of Done

An OCRKit change is complete when:

- responsibility remains recognition-only;
- API compatibility is preserved or versioned;
- field confidence and warning behavior are defined;
- new layouts or fields have regression fixtures;
- production and training dependencies remain separated;
- privacy boundaries are preserved;
- model artifacts are reproducible and immutable;
- tests and evaluation gates pass;
- rollback is documented;
- the consuming platform impact is identified.
