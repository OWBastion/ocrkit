# OCRKit

Overwatch 自定义挑战截图结构化识别服务。

## Stack

- Python 3.12+
- FastAPI
- RapidOCR + ONNX Runtime (default)
- OpenCV
- uv

## Run

```bash
rtk uv sync
rtk uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

- `GET /health`
- `POST /api/v1/ocr/challenge` (`multipart/form-data` with `file`, optional `debug=true`)

## OCR Engine

Default engine is `rapidocr`.

Switch to PaddleOCR (optional dependency):

```bash
rtk uv sync --extra paddle
OCRKIT_OCR_ENGINE=paddleocr rtk uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker Compose

```bash
rtk docker compose up --build -d
rtk docker compose ps
```

## Samoa Integration Check

```bash
rtk uv run pytest tests/test_samoa_integration.py
```
