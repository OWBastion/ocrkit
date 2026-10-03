#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <checkpoint_base> <output_dir> [--holdout-labels <labels.txt> --holdout-images-root <directory>]\n' "$0" >&2
  exit 1
}
if [[ $# -lt 2 ]]; then
  usage
fi
checkpoint="$1"
artifact_dir="$2"
shift 2
holdout_labels=""
holdout_images_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --holdout-labels) [[ $# -ge 2 ]] || usage; holdout_labels="$2"; shift 2 ;;
    --holdout-images-root) [[ $# -ge 2 ]] || usage; holdout_images_root="$2"; shift 2 ;;
    *) usage ;;
  esac
done
if [[ -n "${holdout_labels}" && -z "${holdout_images_root}" ]]; then
  usage
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The comparison needs the model storage settings to find the model in service; without them it would silently
# compare with nothing.
if [[ -f "${root_dir}/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -v "${key}" ]] && continue
    export "${key}=${value}"
  done < "${root_dir}/.env"
fi
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
training_python="${work_dir}/venv/bin/python"
paddle2onnx_bin="${work_dir}/venv/bin/paddle2onnx"
config_path="${paddleocr_dir}/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"

for path in "${paddleocr_dir}" "${training_python}" "${paddle2onnx_bin}" "${checkpoint}.pdparams" "${config_path}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing checkpoint evaluation input: %s\n' "${path}" >&2
    exit 1
  fi
done
if [[ -e "${artifact_dir}" ]]; then
  printf 'checkpoint evaluation output already exists: %s\n' "${artifact_dir}" >&2
  exit 1
fi

mkdir -p "${artifact_dir}/paddle_rec"
cd "${root_dir}"
uv run python training/scripts/prepare_detector.py \
  --lock training/configs/pp_ocrv6_small_det.lock.json \
  --cache-dir "${work_dir}/cache/detector" \
  --output "${artifact_dir}/det.onnx"

cd "${paddleocr_dir}"
"${training_python}" tools/export_model.py -c "${config_path}" -o \
  Global.pretrained_model="${checkpoint}" \
  Global.save_inference_dir="${artifact_dir}/paddle_rec" \
  Global.character_dict_path="${paddleocr_dir}/ppocr/utils/dict/ppocrv6_dict.txt"
"${paddle2onnx_bin}" \
  --model_dir "${artifact_dir}/paddle_rec" \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file "${artifact_dir}/rec.onnx" \
  --opset_version 12
cp "${paddleocr_dir}/ppocr/utils/dict/ppocrv6_dict.txt" "${artifact_dir}/rec_dict.txt"

cd "${root_dir}"
uv run python training/scripts/prepare_rapidocr_config.py \
  --template "${root_dir}/configs/ocr.yaml" \
  --artifact-dir "${artifact_dir}" \
  --output "${artifact_dir}/rapidocr.yaml"
compare_args=(--candidate-artifact "${artifact_dir}")
if [[ -n "${holdout_labels}" ]]; then
  compare_args+=(--holdout-labels "${holdout_labels}" --holdout-images-root "${holdout_images_root}")
fi
uv run python training/scripts/compare_candidate.py "${compare_args[@]}"
