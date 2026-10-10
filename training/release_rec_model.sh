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
# The compatibility gate loads the model in service; the service's default cache is a production path.
export OCRKIT_MODEL_CACHE_DIR="${OCRKIT_MODEL_CACHE_DIR:-${work_dir}/cache/model-in-service}"
training_python="${work_dir}/venv/bin/python"
paddle2onnx_bin="${work_dir}/venv/bin/paddle2onnx"
checkpoint="${work_dir}/checkpoints/rec_pp_ocrv6_small/best_accuracy"
release_channel="${OCRKIT_MODEL_CANDIDATE_CHANNEL_KEY:-models/pp-ocrv6-small/channels/candidate.json}"
evaluated_dir=""
holdout_labels=""
holdout_images_root=""
provenance=""
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
    --release-channel)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>] [--release-channel <channel_key>]\n' "$0" >&2
        exit 2
      fi
      release_channel="$2"
      shift 2
      ;;
    --evaluated-dir)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--evaluated-dir <directory from a passed checkpoint evaluation>] [--provenance <provenance.json>]\n' "$0" >&2
        exit 2
      fi
      evaluated_dir="$2"
      shift 2
      ;;
    --holdout-labels)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>] [--holdout-labels <labels.txt>] [--holdout-images-root <directory>] [--provenance <provenance.json>]\n' "$0" >&2
        exit 2
      fi
      holdout_labels="$2"
      shift 2
      ;;
    --holdout-images-root)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>] [--holdout-labels <labels.txt>] [--holdout-images-root <directory>] [--provenance <provenance.json>]\n' "$0" >&2
        exit 2
      fi
      holdout_images_root="$2"
      shift 2
      ;;
    --provenance)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>] [--holdout-report <report.json>] [--provenance <provenance.json>]\n' "$0" >&2
        exit 2
      fi
      provenance="$2"
      shift 2
      ;;
    *)
      printf 'usage: %s [--checkpoint <checkpoint_base> | --evaluated-dir <directory>] [--provenance <provenance.json>]\n' "$0" >&2
      exit 2
      ;;
  esac
done
bucket="${OCRKIT_R2_DEFAULT_BUCKET:?set OCRKIT_R2_DEFAULT_BUCKET}"
if [[ "${release_channel}" == "models/pp-ocrv6-small/channels/stable.json" ]]; then
  printf 'stable channel requires explicit promotion; publish a candidate instead\n' >&2
  exit 2
fi
if [[ -n "${holdout_labels}" && -z "${holdout_images_root}" ]]; then
  printf '--holdout-images-root is required with --holdout-labels\n' >&2
  exit 2
fi

release_inputs=("${training_python}" "${paddle2onnx_bin}" "${checkpoint}.pdparams")
if [[ -n "${evaluated_dir}" ]]; then
  release_inputs=("${evaluated_dir}/comparison_report.json" "${evaluated_dir}/rec.onnx" "${evaluated_dir}/rapidocr.yaml")
fi
for path in "${release_inputs[@]}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing release input: %s\n' "${path}" >&2
    exit 1
  fi
done

cd "${root_dir}"
version="$(uv run python training/scripts/next_model_version.py --bucket "${bucket}")"
artifact_dir="${work_dir}/artifacts/${version}"
if [[ -n "${evaluated_dir}" ]]; then
  # The checkpoint was already exported and compared with the model in service; do not repeat that work.
  cp -R "${evaluated_dir}" "${artifact_dir}"
else
  evaluation_args=()
  if [[ -n "${holdout_labels}" ]]; then
    evaluation_args+=(--holdout-labels "${holdout_labels}" --holdout-images-root "${holdout_images_root}")
  fi
  "${root_dir}/training/evaluate_rec_checkpoint.sh" "${checkpoint}" "${artifact_dir}" ${evaluation_args[@]+"${evaluation_args[@]}"}
fi
holdout_report=""
if [[ -f "${artifact_dir}/holdout_report.json" ]]; then
  holdout_report="${artifact_dir}/holdout_report.json"
fi

test_report="${artifact_dir}/full_test_report.txt"
set +e
OCRKIT_MODEL_MANIFEST_KEY= uv run pytest -q >"${test_report}" 2>&1
test_status=$?
set -e
cat "${test_report}"
if [[ "${test_status}" -ne 0 ]]; then
  printf 'full test suite failed; see %s\n' "${test_report}" >&2
  exit "${test_status}"
fi
compatibility_report="${artifact_dir}/compatibility_report.json"
uv run python scripts/compatibility_gate.py --report "${compatibility_report}"
evidence_args=(
  --fixture-report "${artifact_dir}/fixture_report.json"
  --comparison-report "${artifact_dir}/comparison_report.json"
  --compatibility-report "${compatibility_report}"
  --output "${artifact_dir}/release_evidence.json"
)
if [[ -n "${holdout_report}" ]]; then
  evidence_args+=(--holdout-report "${holdout_report}")
fi
if [[ -n "${provenance}" ]]; then
  evidence_args+=(--provenance "${provenance}")
fi
uv run python training/scripts/build_release_evidence.py \
  "${evidence_args[@]}"
uv run python training/scripts/build_manifest.py \
  --artifact-dir "${artifact_dir}" \
  --version "${version}" \
  --evidence "${artifact_dir}/release_evidence.json"
uv run python training/scripts/upload_artifacts.py --artifact-dir "${artifact_dir}" --bucket "${bucket}"
uv run python training/scripts/verify_published_artifact.py \
  --bucket "${bucket}" \
  --manifest-key "models/pp-ocrv6-small/${version}/manifest.json"
uv run python training/scripts/publish_model_channel.py \
  --bucket "${bucket}" \
  --channel-key "${release_channel}" \
  --manifest-key "models/pp-ocrv6-small/${version}/manifest.json"

printf 'model_version=%s\n' "${version}"
printf 'published_manifest_key=models/pp-ocrv6-small/%s/manifest.json\n' "${version}"
printf 'release_channel_key=%s\n' "${release_channel}"
printf 'promotion_action=compare the candidate against stable, then run promote_model_channel.py\n'
printf 'deployment_action=recreate the OCRKit container; no per-release env update is required\n'
