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

Run the controlled rollout from the deployment host with a public, non-private
smoke image:

```sh
uv run python training/scripts/production_rollout.py --smoke-image ./smoke.png
```

The command reads the stable channel and manifest before recreation, runs
`docker compose up --no-build --force-recreate`, waits for health, verifies the
exact loaded `model_version` and unchanged manifest hash, and runs an authenticated
local smoke request. It prints only the version, manifest identity, and pass/fail
summary. If it fails, select a previous verified manifest with
`training/scripts/rollback_model_channel.py` and rerun the same rollout command.

After deployment, verify `https://ocr.owbastion.com/health` anonymously. Recognition
endpoints require `Authorization: Bearer <OCRKIT_API_TOKEN>` and are called only by the
platform Worker. Do not place the service token, evidence keys, screenshots, or OCR payloads
in shell history or logs.
