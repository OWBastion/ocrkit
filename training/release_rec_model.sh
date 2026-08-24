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
release_channel="${OCRKIT_MODEL_CANDIDATE_CHANNEL_KEY:-models/pp-ocrv6-small/channels/candidate.json}"
holdout_report=""
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
    --holdout-report)
      if [[ $# -lt 2 ]]; then
        printf 'usage: %s [--checkpoint <checkpoint_base>] [--holdout-report <report.json>] [--provenance <provenance.json>]\n' "$0" >&2
        exit 2
      fi
      holdout_report="$2"
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
      printf 'usage: %s [--checkpoint <checkpoint_base>] [--holdout-report <report.json>] [--provenance <provenance.json>]\n' "$0" >&2
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

if [[ -n "${holdout_labels}" ]]; then
  uv run python training/scripts/evaluate_rec_holdout.py \
    --labels "${holdout_labels}" \
    --images-root "${holdout_images_root}" \
    --model-config "${artifact_dir}/rapidocr.yaml" \
    --report "${artifact_dir}/holdout_report.json"
elif [[ -n "${holdout_report}" ]]; then
  if [[ ! -f "${holdout_report}" ]]; then
    printf 'missing holdout report: %s\n' "${holdout_report}" >&2
    exit 1
  fi
  cp "${holdout_report}" "${artifact_dir}/holdout_report.json"
fi
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
evidence_args=(--fixture-report "${artifact_dir}/fixture_report.json" --output "${artifact_dir}/release_evidence.json")
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
