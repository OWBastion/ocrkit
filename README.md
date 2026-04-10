# Bastion OCR Service

FastAPI + PaddleOCR 服务，用于从 Bastion 通关截图识别通关信息并计算可自动判定的称号。

## API

- `GET /ping`
  - 返回服务可用状态与称号规则版本。
- `POST /extract`
  - 请求：
    ```json
    { "imageUrl": "https://..." }
    ```
  - 返回：
    - `extracted`: `passed/playerName/timeSec/deaths/skips/mapLabel/difficulty/ocrTexts`
    - `titleDecision`: `awardedKeys/notAwardedKeys/notEvaluatedKeys/reasons/confidence`

## Run

```bash
rtk pip install -e .[dev]
rtk proxy uvicorn main:app --host 0.0.0.0 --port 8000
```

## Rules Source

默认从以下地址拉取称号规则并缓存 10 分钟：

- https://raw.githubusercontent.com/OWBastion/Bastion/main/data/title-source.json
