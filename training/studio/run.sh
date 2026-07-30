#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
frontend_dir="${root_dir}/training/studio/frontend"

build_frontend() {
  pnpm --dir "${frontend_dir}" install --frozen-lockfile
  pnpm --dir "${frontend_dir}" run build
}

usage() {
  printf 'usage: %s [build|start] [Studio server options]\n' "$0"
  printf '  build       install locked frontend dependencies and build the UI\n'
  printf '  start       build the UI, then start Studio on 127.0.0.1:7860 (default)\n'
}

command="${1:-start}"
case "${command}" in
  build)
    if [[ $# -ne 1 ]]; then
      usage >&2
      exit 2
    fi
    build_frontend
    ;;
  start)
    if [[ $# -gt 0 ]]; then
      shift
    fi
    build_frontend
    exec uv run --extra vision python -m training.studio "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
