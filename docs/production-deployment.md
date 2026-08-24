# Production deployment

Deploy OCRKit on `US-LOS-C-1` with `docker compose -f docker-compose.production.yml --env-file .env up -d`.
Use a GHCR image pinned by digest. Configure a Cloudflare Tunnel public hostname for
`ocr.owbastion.com` whose service is `http://ocrkit:8000`; do not publish a host port.

The deployment `.env` is derived from `.env.production.example` and must remain on the host.
Its R2 credential is read-only and may access only the OCRKit model bucket plus
`owbastion-codes-evidence`. `OCRKIT_ALLOW_DEBUG` is forced off in production.

The production Compose file defaults
`OCRKIT_MODEL_RELEASE_CHANNEL_KEY` to `models/pp-ocrv6-small/channels/stable.json`. Keep
`OCRKIT_MODEL_MANIFEST_KEY` only as an optional legacy rollback target; the channel takes
precedence when both are set. After a verified Studio candidate is explicitly promoted (or a rollback
selects a prior verified manifest), recreate the OCRKit container to download and verify the selected
model before serving traffic. No per-release server environment edit is required.

After deployment, verify `https://ocr.owbastion.com/health` anonymously. Recognition
endpoints require `Authorization: Bearer <OCRKIT_API_TOKEN>` and are called only by the
platform Worker. Do not place the service token, evidence keys, screenshots, or OCR payloads
in shell history or logs.
