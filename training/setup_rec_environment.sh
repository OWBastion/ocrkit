#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
python_bin="${work_dir}/venv/bin/python"
pretrained_dir="${work_dir}/pretrained"
pretrained_model="${pretrained_dir}/PP-OCRv6_small_rec_pretrained.pdparams"

"${root_dir}/training/bootstrap.sh"
"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install "paddlepaddle==3.3.1"
"${python_bin}" -m pip install -r "${paddleocr_dir}/requirements.txt"

mkdir -p "${pretrained_dir}"
if [[ ! -f "${pretrained_model}" ]]; then
  "${python_bin}" -c "from urllib.request import urlretrieve; urlretrieve('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams', '${pretrained_model}')"
fi
