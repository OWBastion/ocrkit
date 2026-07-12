#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <checkpoint_base> <output_dir>\n' "$0" >&2
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
training_python="${work_dir}/venv/bin/python"
paddle2onnx_bin="${work_dir}/venv/bin/paddle2onnx"
checkpoint="$1"
artifact_dir="$2"
config_path="${paddleocr_dir}/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"
min_field_accuracy="0.9604221635883905"

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
OCRKIT_MODEL_MANIFEST_KEY= uv run python scripts/batch_eval.py \
  --model-config "${artifact_dir}/rapidocr.yaml" \
  --report "${artifact_dir}/fixture_report.json" \
  --min-field-accuracy "${min_field_accuracy}"
