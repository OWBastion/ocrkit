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
