#!/usr/bin/env bash
set -euo pipefail

digest=${1:?Image digest is required}
compose_file=${2:?Compose file is required}
[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]
cd /opt/ocrkit
image="ghcr.io/owbastion/ocrkit@$digest"
export OCRKIT_IMAGE="$image"
compose=(docker compose -f "$compose_file" --env-file .env)

# Refuse a host configuration that ignores the pinned image variable.
[[ "$("${compose[@]}" config --images ocrkit)" == "$image" ]]
docker pull "$image"
"${compose[@]}" up -d --no-build --no-deps --force-recreate --wait --wait-timeout 300 ocrkit
container=$("${compose[@]}" ps -q ocrkit)
[[ -n "$container" ]]
[[ "$(docker inspect --format '{{.Config.Image}}' "$container")" == "$image" ]]
docker exec "$container" python -c 'import json; from urllib.request import urlopen; data = json.load(urlopen("http://127.0.0.1:8000/health", timeout=10)); assert data["ok"] is True; print("Loaded model:", data["model_version"])'
# Preserve the digest for subsequent manual restarts without exposing .env.
python3 - "$image" <<'PYENV'
from pathlib import Path
import sys

path = Path('.env')
lines = path.read_text().splitlines()
lines = [line for line in lines if not line.startswith('OCRKIT_IMAGE=')]
lines.append('OCRKIT_IMAGE=' + sys.argv[1])
path.write_text('\n'.join(lines) + '\n')
PYENV
printf 'Deployed %s\n' "$image"
