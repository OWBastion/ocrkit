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
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

- `GET /health`
- `POST /api/v1/ocr/challenge` (`multipart/form-data` with `file`, optional `debug=true`)
- `POST /api/v1/ocr/challenge/by-object` (`application/json` with `object_key`, optional `bucket`, `version_id`, `debug`)

### R2 Object Mode

Configure these environment variables to enable `by-object` endpoint:

- `OCRKIT_R2_ENDPOINT_URL`
- `OCRKIT_R2_ACCESS_KEY_ID`
- `OCRKIT_R2_SECRET_ACCESS_KEY`
- `OCRKIT_R2_REGION_NAME` (default: `auto`)
- `OCRKIT_R2_DEFAULT_BUCKET`
- `OCRKIT_R2_ALLOWED_BUCKETS` (comma-separated whitelist)
- `OCRKIT_R2_READ_TIMEOUT_SECONDS` (default: `10`)

## OCR Engine

Default engine is `rapidocr`.

Switch to PaddleOCR (optional dependency):

```bash
uv sync --extra paddle
OCRKIT_OCR_ENGINE=paddleocr uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker Compose

```bash
docker compose up --build -d
docker compose ps
```

## Samoa Integration Check

```bash
uv run pytest tests/test_samoa_integration.py
```
