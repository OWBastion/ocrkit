#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
frontend_dir="${root_dir}/training/studio/frontend"

load_dotenv() {
  if [[ -f "${root_dir}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${root_dir}/.env"
    set +a
  fi
}

install_frontend() {
  pnpm --dir "${frontend_dir}" install --frozen-lockfile
}

build_frontend() {
  install_frontend
  pnpm --dir "${frontend_dir}" run build
}

port_from_args() {
  local previous=""
  for arg in "$@"; do
    if [[ "${previous}" == "--port" ]]; then
      printf '%s' "${arg}"
      return
    fi
    previous="${arg}"
  done
  printf '7860'
}

refuse_if_running() {
  local port="$1" health
  if health="$(curl -fsS --max-time 2 "http://127.0.0.1:${port}/api/health" 2>/dev/null)"; then
    printf 'Studio is already serving port %s:\n  %s\nStop it first, or pass --port.\n' "${port}" "${health}" >&2
    exit 1
  fi
}

start_development() {
  load_dotenv
  install_frontend
  uv run --extra vision python -m training.studio --port 7860 --api-only --reload &
  api_pid="$!"
  trap 'kill "${api_pid}" 2>/dev/null || true' EXIT INT TERM
  pnpm --dir "${frontend_dir}" run dev
}

usage() {
  printf 'usage: %s [Studio server options]   build the UI and serve everything on http://127.0.0.1:7860\n' "$0"
  printf '       %s --host 0.0.0.0           also reachable from a phone on the same network\n' "$0"
  printf '       %s build                    install locked frontend dependencies and build the UI only\n' "$0"
  printf '       %s dev                      API + Vite HMR on http://127.0.0.1:5173 (UI development)\n' "$0"
}

command="${1:-serve}"
case "${command}" in
  build)
    if [[ $# -ne 1 ]]; then
      usage >&2
      exit 2
    fi
    build_frontend
    ;;
  dev)
    if [[ $# -ne 1 ]]; then
      usage >&2
      exit 2
    fi
    refuse_if_running 7860
    start_development
    ;;
  -h|--help|help)
    usage
    ;;
  serve|start|-*)
    if [[ $# -gt 0 && "${command}" != -* ]]; then
      shift
    fi
    refuse_if_running "$(port_from_args "$@")"
    load_dotenv
    build_frontend
    exec uv run --extra vision python -m training.studio "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
