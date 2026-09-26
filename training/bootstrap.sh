#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"

mkdir -p "${work_dir}"

if [[ ! -d "${paddleocr_dir}/.git" ]]; then
  git clone --branch release/3.7 --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git "${paddleocr_dir}"
fi

printf 'PaddleOCR checkout: %s\n' "${paddleocr_dir}"

venv_dir="${work_dir}/venv"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  if command -v python3.12 >/dev/null && python3.12 -m venv "${venv_dir}"; then
    :
  elif command -v uv >/dev/null; then
    # Managed runtimes such as Colab ship python3.12 without python3-venv/ensurepip.
    uv venv --clear --seed --python 3.12 "${venv_dir}"
  else
    printf 'Python 3.12 with venv support, or uv, is required to create the training environment.\n' >&2
    exit 1
  fi
fi

printf 'Training environment: %s\n' "${venv_dir}"
printf 'Install PaddlePaddle and the PaddleOCR training requirements in that environment before running training/run_rec_smoke.sh.\n'
