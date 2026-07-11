#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <version>\n' "$0" >&2
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'invalid release version: %s\n' "$1" >&2
  exit 1
fi

work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
training_python="${work_dir}/venv/bin/python"
paddle2onnx_bin="${work_dir}/venv/bin/paddle2onnx"
checkpoint="${work_dir}/checkpoints/rec_pp_ocrv6_small/best_accuracy"
artifact_dir="${work_dir}/artifacts/$1"
det_model="${OCRKIT_RELEASE_DET_MODEL:?set OCRKIT_RELEASE_DET_MODEL to the fixed PP-OCRv6 small det.onnx path}"
bucket="${OCRKIT_MODEL_R2_BUCKET:?set OCRKIT_MODEL_R2_BUCKET}"
config_path="${paddleocr_dir}/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"

for path in "${paddleocr_dir}" "${training_python}" "${paddle2onnx_bin}" "${checkpoint}.pdparams" "${det_model}" "${config_path}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing release input: %s\n' "${path}" >&2
    exit 1
  fi
done

if [[ -e "${artifact_dir}" ]]; then
  printf 'release artifact directory already exists: %s\n' "${artifact_dir}" >&2
  exit 1
fi

mkdir -p "${artifact_dir}/paddle_rec"

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

cp "${det_model}" "${artifact_dir}/det.onnx"
cp "${paddleocr_dir}/ppocr/utils/dict/ppocrv6_dict.txt" "${artifact_dir}/rec_dict.txt"
cp "${root_dir}/configs/ocr.yaml" "${artifact_dir}/rapidocr.yaml"

cd "${root_dir}"
uv run pytest -q
OCRKIT_MODEL_MANIFEST_KEY= uv run python scripts/batch_eval.py \
  --model-config "${artifact_dir}/rapidocr.yaml" \
  --min-field-accuracy 0.9656992084432717
uv run python training/scripts/build_manifest.py --artifact-dir "${artifact_dir}" --version "$1"
uv run python training/scripts/upload_artifacts.py --artifact-dir "${artifact_dir}" --bucket "${bucket}"
uv run python training/scripts/verify_published_artifact.py \
  --bucket "${bucket}" \
  --manifest-key "models/pp-ocrv6-small/$1/manifest.json"
