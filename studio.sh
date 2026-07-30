#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="${root_dir}/training/studio/frontend"

install_frontend() {
  pnpm --dir "${frontend_dir}" install --frozen-lockfile
}

build_frontend() {
  install_frontend
  pnpm --dir "${frontend_dir}" run build
}

start_development() {
  install_frontend
  uv run --extra vision python -m training.studio --port 7860 --api-only --reload &
  api_pid="$!"
  trap 'kill "${api_pid}" 2>/dev/null || true' EXIT INT TERM
  pnpm --dir "${frontend_dir}" run dev
}

usage() {
  printf 'usage: %s [build|start|dev] [Studio server options]\n' "$0"
  printf '  build       install locked frontend dependencies and build the UI\n'
  printf '  start       build the UI, then start Studio on http://127.0.0.1:7860\n'
  printf '  dev         run the API and Vite HMR frontend on http://127.0.0.1:5173 (default)\n'
}

command="${1:-dev}"
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
  dev)
    if [[ $# -gt 1 ]]; then
      usage >&2
      exit 2
    fi
    start_development
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
