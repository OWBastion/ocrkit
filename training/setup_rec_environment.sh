#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
python_bin="${work_dir}/venv/bin/python"
pretrained_dir="${work_dir}/pretrained"
pretrained_model="${pretrained_dir}/PP-OCRv6_small_rec_pretrained.pdparams"

bash "${root_dir}/training/bootstrap.sh"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null; then
    printf 'Homebrew is required to install ccache on macOS. Install Homebrew, then rerun this script.\n' >&2
    exit 1
  fi
  if ! command -v ccache >/dev/null; then
    brew install ccache
  fi
  if ! command -v ccache >/dev/null; then
    printf 'ccache installation completed but ccache is not on PATH.\n' >&2
    exit 1
  fi
fi

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install "paddlepaddle==3.3.1" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
"${python_bin}" -m pip install -r "${paddleocr_dir}/requirements.txt"

if [[ "$(uname -s)" == "Darwin" ]]; then
  "${python_bin}" -c 'import platform; import paddle; assert platform.machine() == "arm64"; assert paddle.device.get_device() == "cpu"; assert not paddle.is_compiled_with_cuda(); print(f"PaddlePaddle {paddle.__version__}: {platform.machine()} {paddle.device.get_device()}")'
fi

mkdir -p "${pretrained_dir}"
if [[ ! -f "${pretrained_model}" ]]; then
  "${python_bin}" -c "from urllib.request import urlretrieve; urlretrieve('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams', '${pretrained_model}')"
fi
