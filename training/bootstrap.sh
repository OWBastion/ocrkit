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
  python3.12 -m venv "${venv_dir}"
fi

printf 'Training environment: %s\n' "${venv_dir}"
printf 'Install PaddlePaddle and the PaddleOCR training requirements in that environment before running training/run_rec_smoke.sh.\n'
