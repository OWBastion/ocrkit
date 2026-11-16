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

## Asynchronous OCR delivery

Deploy OCRKit with the jobs endpoint first; existing synchronous callers keep working.
Then apply the platform migration and deploy its callback receiver and jobs dispatcher
together. The receiver must be ready before the platform dispatches a new job.
The existing service token authenticates both admission and callbacks. Configure
`OCRKIT_PLATFORM_BASE_URL` to the platform HTTPS origin and retain the `ocrkit-jobs`
volume across restarts. Use one service replica and one Uvicorn worker per spool.
Register the callback operation in the platform API Shield configuration. If the
zone blocks the deployment host's region, extend the existing OCR service rule
to allow only that host's source IP, the platform hostname, POST, and the
`/v1/ocrkit/jobs/{jobId}/result` path. Keep Bearer authentication enabled in the
receiver; recognition intake remains restricted to the existing Worker callers.
The spool contains private screenshots until inference completes, results until the
platform acknowledges, and ID hashes for up to 24 hours. Do not copy it into training
artifacts or public backups. Shutdown waits for current synchronous inference; it
cannot safely hard-cancel the engine. See README for deadlines, admission limits,
retry behavior, and restart semantics.

## GitHub Actions application deployment

A push to `main` publishes the application image. After both the existing Python
checks and image publication succeed, the `Docker GHCR` workflow deploys that
build's digest over SSH. PRs and `v*` tags do not deploy. A run skips deployment
if `main` has advanced before deployment starts. Production deployments run one
at a time; a later build does not cancel an active deployment.

Configure the repository's GitHub `production` environment:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `DEPLOY_SSH_KEY` | Dedicated deployment SSH private key; install its public key on the server first |
| Secret | `DEPLOY_KNOWN_HOSTS` | OpenSSH known_hosts records verified through a trusted channel; use `[host]:port` for non-default ports |
| Variable | `DEPLOY_HOST` | Deployment server address |
| Variable | `DEPLOY_USER` | SSH user with Docker access and permission to update the host `.env` |
| Variable | `DEPLOY_PORT` | SSH port; defaults to `22` |
| Variable | `DEPLOY_COMPOSE_FILE` | Existing Compose filename inside `/opt/ocrkit` |

The server must have Docker Compose v2 with `--wait` support, Python 3, the
existing `/opt/ocrkit/.env`, and the production Compose configuration. Configure
read-only GHCR login on the server if the image is private. Runtime credentials
stay on the server. Do not replace the host's network/tunnel configuration with
the repository example. Select `docker-compose.production.yml` for the topology
above; an installation using an existing tunnel must select its actual host
Compose file, such as `docker-compose.existing-tunnel.yml`.

The host Compose service must be named `ocrkit` and use `${OCRKIT_IMAGE}`.
The script rejects a configuration that ignores the supplied digest before
pulling or restarting. It recreates only `ocrkit`, preserves model/jobs volumes
and the tunnel, waits for Compose readiness, checks the exact container image,
and checks `/health` inside the container. It reports the loaded model version
and saves the successful digest as `OCRKIT_IMAGE` in the host `.env`.
A failure marks the job failed; it does not automatically roll back.

This deploys application code. It does not promote or roll back a model channel.
The model rollout procedure above remains a separate operation. The automated
health check does not prove recognition accuracy or public tunnel reachability.
After first enabling deployment, verify public health and authenticated OCR with
a public smoke image. To roll back application code, set the previous verified
`OCRKIT_IMAGE` digest in the host `.env`, then run
`docker compose -f <host Compose file> --env-file .env up -d --no-build --no-deps --force-recreate --wait ocrkit`.
