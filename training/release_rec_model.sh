#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${root_dir}/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -v "${key}" ]] && continue
    export "${key}=${value}"
  done < "${root_dir}/.env"
fi

work_dir="${root_dir}/training/.work"
training_python="${work_dir}/venv/bin/python"
paddle2onnx_bin="${work_dir}/venv/bin/paddle2onnx"
checkpoint="${work_dir}/checkpoints/rec_pp_ocrv6_small/best_accuracy"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>]\n' "$0" >&2
        exit 2
      fi
      checkpoint="$2"
      shift 2
      ;;
    *)
      printf 'usage: %s [--checkpoint <checkpoint_base>]\n' "$0" >&2
      exit 2
      ;;
  esac
done
bucket="${OCRKIT_R2_DEFAULT_BUCKET:?set OCRKIT_R2_DEFAULT_BUCKET}"

for path in "${training_python}" "${paddle2onnx_bin}" "${checkpoint}.pdparams"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing release input: %s\n' "${path}" >&2
    exit 1
  fi
done

cd "${root_dir}"
version="$(uv run python training/scripts/next_model_version.py --bucket "${bucket}")"
artifact_dir="${work_dir}/artifacts/${version}"
"${root_dir}/training/evaluate_rec_checkpoint.sh" "${checkpoint}" "${artifact_dir}"

OCRKIT_MODEL_MANIFEST_KEY= uv run pytest -q
uv run python training/scripts/build_manifest.py --artifact-dir "${artifact_dir}" --version "${version}"
uv run python training/scripts/upload_artifacts.py --artifact-dir "${artifact_dir}" --bucket "${bucket}"
uv run python training/scripts/verify_published_artifact.py \
  --bucket "${bucket}" \
  --manifest-key "models/pp-ocrv6-small/${version}/manifest.json"

printf 'model_version=%s\n' "${version}"
printf 'manifest_key=models/pp-ocrv6-small/%s/manifest.json\n' "${version}"
printf 'OCRKIT_MODEL_MANIFEST_KEY=models/pp-ocrv6-small/%s/manifest.json\n' "${version}"
