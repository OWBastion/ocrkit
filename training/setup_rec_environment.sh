#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
python_bin="${work_dir}/venv/bin/python"
pretrained_dir="${work_dir}/pretrained"
pretrained_model="${pretrained_dir}/PP-OCRv6_small_rec_pretrained.pdparams"

device=cpu
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--device" ]]; then
    printf 'usage: %s [--device cpu|cuda]\n' "$0" >&2
    exit 2
  fi
  device="$2"
fi
if [[ "${device}" != cpu && "${device}" != cuda ]]; then
  printf 'device must be cpu or cuda\n' >&2
  exit 2
fi

bash "${root_dir}/training/bootstrap.sh"

if [[ "${device}" == cpu && "$(uname -s)" == "Darwin" ]]; then
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

paddle_package="paddlepaddle==3.3.1"
paddle_index="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
if [[ "${device}" == cuda ]]; then
  if [[ "$(uname -s)" != Linux ]] || ! command -v nvidia-smi >/dev/null; then
    printf 'CUDA training requires a Linux runtime with an NVIDIA GPU and nvidia-smi.\n' >&2
    exit 1
  fi

  cuda_version="$(nvidia-smi 2>/dev/null | awk -F 'CUDA Version: ' 'NF > 1 {split($2, version, " "); print version[1]; exit}')"
  if [[ -z "${cuda_version}" ]]; then
    printf 'could not determine the NVIDIA driver CUDA version from nvidia-smi.\n' >&2
    exit 1
  fi
  cuda_major="$(cut -d. -f1 <<< "${cuda_version}")"
  cuda_minor="$(cut -d. -f2 <<< "${cuda_version}")"
  if (( cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 9) )); then
    paddle_index="https://www.paddlepaddle.org.cn/packages/stable/cu129/"
  elif (( cuda_major == 12 && cuda_minor >= 6 )); then
    paddle_index="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
  elif (( cuda_major == 11 && cuda_minor >= 8 )) || (( cuda_major == 12 )); then
    paddle_index="https://www.paddlepaddle.org.cn/packages/stable/cu118/"
  else
    printf 'NVIDIA CUDA %s is unsupported; PaddlePaddle 3.3.1 needs CUDA 11.8 or newer.\n' "${cuda_version}" >&2
    exit 1
  fi

  paddle_package="paddlepaddle-gpu==3.3.1"
fi

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install "${paddle_package}" -i "${paddle_index}"
"${python_bin}" -m pip install -r "${paddleocr_dir}/requirements.txt"
"${python_bin}" -m pip install "paddle2onnx==2.1.0"

if [[ "${device}" == cpu && "$(uname -s)" == "Darwin" ]]; then
  "${python_bin}" -c 'import platform; import paddle; assert platform.machine() == "arm64"; assert paddle.device.get_device() == "cpu"; assert not paddle.is_compiled_with_cuda(); print(f"PaddlePaddle {paddle.__version__}: {platform.machine()} {paddle.device.get_device()}")'
fi

if [[ "${device}" == cuda ]]; then
  "${python_bin}" -c 'import paddle; assert paddle.is_compiled_with_cuda(), "installed PaddlePaddle is not CUDA-enabled"; assert paddle.device.cuda.device_count() > 0, "no CUDA device is visible to PaddlePaddle"; paddle.device.set_device("gpu:0"); assert paddle.to_tensor([1.0]).numpy()[0] == 1.0; print(f"PaddlePaddle {paddle.__version__}: CUDA {paddle.device.get_device()}")'
fi

mkdir -p "${pretrained_dir}"
if [[ ! -f "${pretrained_model}" ]]; then
  "${python_bin}" -c "from urllib.request import urlretrieve; urlretrieve('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams', '${pretrained_model}')"
fi
