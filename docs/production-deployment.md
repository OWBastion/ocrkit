# Production deployment

Deploy OCRKit on `US-LOS-C-1` with `docker compose -f docker-compose.production.yml --env-file .env up -d`.
Use a GHCR image pinned by digest. Configure a Cloudflare Tunnel public hostname for
`ocr.owbastion.com` whose service is `http://ocrkit:8000`; do not publish a host port.

The deployment `.env` is derived from `.env.production.example` and must remain on the host.
Its R2 credential is read-only and may access only the OCRKit model bucket plus
`owbastion-codes-evidence`. `OCRKIT_ALLOW_DEBUG` is forced off in production.

After deployment, verify `https://ocr.owbastion.com/health` anonymously. Recognition
endpoints require `Authorization: Bearer <OCRKIT_API_TOKEN>` and are called only by the
platform Worker. Do not place the service token, evidence keys, screenshots, or OCR payloads
in shell history or logs.
