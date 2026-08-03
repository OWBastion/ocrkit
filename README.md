# OCRKit

OCR Api for [Bastion](https://github.com/OWBastion/Bastion)

## Stack

- Python 3.12+
- FastAPI
- RapidOCR + ONNX Runtime (default)
- OpenCV
- uv

## Run

```bash
git clone --recurse-submodules git@github.com:OWBastion/ocrkit.git
cd ocrkit
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For an existing checkout, initialize the private datasets submodule before running
tests or training commands:

```bash
git submodule update --init --recursive
```

## API

- `GET /health`
- `POST /api/v1/ocr/challenge` (`multipart/form-data` with `file`, optional `debug=true`)
- `POST /api/v1/ocr/challenge/by-object` (`application/json` with `object_key`, optional `bucket`, `version_id`, `debug`)

Set `OCRKIT_AGENTS_API_BASE_URL` to the platform origin to load active title labels from
`/v1/agents/titles` at startup. `OCRKIT_AGENTS_API_TIMEOUT_SECONDS` bounds that request.

Successful recognition responses retain the `data`, `warnings`, and optional `debug` fields
and also include traceable contract metadata:

```json
{
  "schema_version": "1",
  "request_id": "...",
  "engine": "rapidocr",
  "model_version": "builtin",
  "layout_version": "1280x720-v5",
  "ok": true,
  "data": {},
  "fields": {
    "viewer_player": {
      "value": "...",
      "confidence": 0.98,
      "source_roi": ["bottom_left_hero"],
      "normalization": [],
      "status": "ok"
    }
  },
  "warnings": [],
  "quality": {
    "original_size": [2560, 1440],
    "aspect_ratio": 1.7778,
    "layout_confidence": 1.0,
    "cropped": false,
    "blur_score": 0.08,
    "normalized_size": [1280, 720],
    "layout_version": "1280x720-v5",
    "warnings": []
  }
}
```

`quality` describes input evidence before normalization. `aspect_ratio` is rounded to four
decimal places; `cropped` and the related warnings indicate a ratio mismatch against the
configured layout and do not make an approval decision. `blur_score` is a normalized blur
risk from 0 to 1, where higher means blurrier.

Send `X-Request-ID` to correlate a request; otherwise OCRKit generates one. Each field
reports its parsed value, confidence, source ROI, normalization metadata, and status.
`achievement_panel_text` contains raw OCR evidence from the dedicated left-side achievement
panel ROI. It may contain the panel's progress and statistics in addition to the challenge
text; the platform should match it against its current challenge/title catalog and decide
whether the evidence is adopted or grants anything. `achievement_titles` contains matching
titles from the platform Agents catalog detected in the left-panel ROIs, in top-to-bottom
order. `achievement_title` is the first title for compatibility, and
`achievement_unlocked` is `true` when at least one matching title is followed by a visible
completion checkmark. The player name in the lower-left hero panel is not used for this
signal. If the ROI or catalog cannot be read, the title fields are empty/`null`.
`viewer_player` is the only player identity field suitable for account matching. The
central completion banner is not an authoritative player source, so its player name is
not included in `ChallengeData` or field evidence.
`map_variant` is `"classic"` when the right-panel map label contains `经典版` or `经典`,
and `null` when the variant is not detected.

### R2 Object Mode

Configure these environment variables to enable `by-object` endpoint:

- `OCRKIT_R2_ENDPOINT_URL`
- `OCRKIT_R2_ACCESS_KEY_ID`
- `OCRKIT_R2_SECRET_ACCESS_KEY`
- `OCRKIT_R2_REGION_NAME` (default: `auto`)
- `OCRKIT_R2_DEFAULT_BUCKET`
- `OCRKIT_R2_ALLOWED_BUCKETS` (comma-separated whitelist)
- `OCRKIT_R2_READ_TIMEOUT_SECONDS` (default: `10`)

For platform evidence, the caller must provide `bucket` explicitly in the
`by-object` request. Add the platform evidence bucket to
`OCRKIT_R2_ALLOWED_BUCKETS`; do not point `OCRKIT_R2_DEFAULT_BUCKET` at the
platform bucket when it is reserved for OCRKit models or other service-owned
objects.

The local OCRKit Studio can use the same read-only R2 credential set to import
screenshot sources without a platform API token. Configure the
Studio-only bucket and prefix allowlist:

- `OCRKIT_STUDIO_R2_BUCKET`
- `OCRKIT_STUDIO_R2_ALLOWED_PREFIXES` (comma-separated, for example `uploads/`)
- `OCRKIT_STUDIO_R2_MAX_OBJECTS` (default: `200`)
- `OCRKIT_STUDIO_R2_MAX_OBJECT_BYTES` (default: `26214400`)

These settings affect only the local Studio API. Credentials stay server-side;
the browser receives object metadata and the selected images are copied into
the private ignored Studio batch. Use a dedicated R2 access key with read-only
permission scoped to this bucket and prefix. Studio never lists or reads the
production model prefix through this import path.

## OCR Engine and Model Artifacts

Default engine is `rapidocr`. Production RapidOCR loads a versioned PP-OCRv6 small
artifact manifest from Cloudflare R2; the service image does not contain PaddleOCR or
model binaries. It reuses the existing R2 endpoint and credentials, plus:

- `OCRKIT_R2_DEFAULT_BUCKET`
- `OCRKIT_MODEL_MANIFEST_KEY` (the initial versioned fallback manifest)
- `OCRKIT_MODEL_RELEASE_CHANNEL_KEY` (recommended: `models/pp-ocrv6-small/channels/stable.json`)
- `OCRKIT_MODEL_CACHE_DIR` (default: `/var/lib/ocrkit/models`)
- `OCRKIT_MODEL_DOWNLOAD_TIMEOUT_SECONDS` (default: `30`)

When a release channel is configured, it takes precedence over the fallback manifest at container
startup. The service verifies the selected immutable artifact before serving traffic. Without either
model setting, local development uses RapidOCR's bundled default model.

## Rust image preflight

The Rust image core validates the versioned ROI layout manifest, decodes and
normalizes training screenshots, exports raw ROI PNGs, records source/crop
SHA-256 provenance, maps standard-layout coordinates, and reports aspect-ratio
quality warnings. The authoritative layout remains
configs/roi_1280x720.yaml; regenerate and check its JSON manifest with:

~~~bash
uv run python scripts/export_layout_manifest.py \
  configs/roi_1280x720.yaml \
  configs/roi_1280x720.manifest.json \
  --check
cargo test --manifest-path rust/Cargo.toml
~~~

The native CLI participates in Studio training-data preparation only. Python
continues to perform ROI-specific preprocessing, RapidOCR/Vision recognition,
human review, and PaddleOCR training. Rust does not replace the production OCR
path or make approval decisions. See
rust/README.md.

PaddleOCR is only used offline for fine-tuning and export. See
[`training/README.md`](training/README.md) for the PP-OCRv6 small det/rec label formats,
validation, manifest generation, and Cloudflare R2 publication workflow.

```bash
uv run python scripts/batch_eval.py
```

## Docker Compose

For a deployed model, copy `.env.model.example` to a deployment-only `.env`, fill in the R2
credentials, retain a known-good `OCRKIT_MODEL_MANIFEST_KEY` for legacy rollback, and set the stable
`OCRKIT_MODEL_RELEASE_CHANNEL_KEY` once after that channel exists. Models use the reserved `models/pp-ocrv6-small/` prefix;
user screenshots use `uploads/`. Do not commit that file. The container downloads and verifies the
model at startup, then follows the channel without later environment edits.

```bash
docker compose up --build -d
docker compose ps
```

Studio updates the stable channel after a fully verified publication. Restart or recreate the OCRKit
container to adopt a newly published or rolled-back channel target. The old environment-variable flow
remains available for initial migration:

```bash
docker compose up -d --force-recreate
```
