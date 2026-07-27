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

Successful recognition responses retain the `data`, `warnings`, and optional `debug` fields
and also include traceable contract metadata:

```json
{
  "schema_version": "1",
  "request_id": "...",
  "engine": "rapidocr",
  "model_version": "builtin",
  "layout_version": "1280x720-v1",
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
    "layout_version": "1280x720-v1",
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
`achievement_title` is the title text detected in the lower-left hero panel, and
`achievement_unlocked` is `true` when that title is visible in the screenshot. If the
ROI cannot be read, both fields are `null` rather than being treated as `false`.
`viewer_player` is the only player identity field suitable for account matching. The
central completion banner is not an authoritative player source, so its player name is
not included in `ChallengeData` or field evidence.

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

## OCR Engine and Model Artifacts

Default engine is `rapidocr`. Production RapidOCR loads a versioned PP-OCRv6 small
artifact manifest from Cloudflare R2; the service image does not contain PaddleOCR or
model binaries. It reuses the existing R2 endpoint and credentials, plus:

- `OCRKIT_R2_DEFAULT_BUCKET`
- `OCRKIT_MODEL_MANIFEST_KEY` (a versioned `models/pp-ocrv6-small/<version>/manifest.json` key)
- `OCRKIT_MODEL_CACHE_DIR` (default: `/var/lib/ocrkit/models`)
- `OCRKIT_MODEL_DOWNLOAD_TIMEOUT_SECONDS` (default: `30`)

When `OCRKIT_MODEL_MANIFEST_KEY` is configured, a missing, incomplete, or checksum-invalid
model artifact prevents the service from starting. Without it, local development uses
RapidOCR's bundled default model.

PaddleOCR is only used offline for fine-tuning and export. See
[`training/README.md`](training/README.md) for the PP-OCRv6 small det/rec label formats,
validation, manifest generation, and Cloudflare R2 publication workflow.

```bash
uv run python scripts/batch_eval.py
```

## Docker Compose

For a deployed model, copy `.env.model.example` to a deployment-only `.env`, fill in the R2
credentials and set `OCRKIT_MODEL_MANIFEST_KEY` to the released version. Models use the reserved
`models/pp-ocrv6-small/` prefix; user screenshots use `uploads/`. Do not commit that file.
The container downloads and verifies the model at startup, then caches it in the named volume.

```bash
docker compose up --build -d
docker compose ps
```

To switch or roll back, change only `OCRKIT_MODEL_MANIFEST_KEY` and recreate the service:

```bash
docker compose up -d --force-recreate
```
