#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${root_dir}/training/.work"
paddleocr_dir="${work_dir}/PaddleOCR"
python_bin="${work_dir}/venv/bin/python"
labels_dir="${root_dir}/datasets/labeled/rec"
config_path="${paddleocr_dir}/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"
output_dir="${work_dir}/checkpoints/rec_pp_ocrv6_small"

for path in "${paddleocr_dir}" "${python_bin}" "${labels_dir}/labels/train.txt" "${labels_dir}/labels/holdout.txt"; do
  if [[ ! -e "${path}" ]]; then
    printf 'missing required training input: %s\n' "${path}" >&2
    exit 1
  fi
done

"${python_bin}" "${root_dir}/training/scripts/validate_annotations.py" rec "${labels_dir}/labels/train.txt"
"${python_bin}" "${root_dir}/training/scripts/validate_annotations.py" rec "${labels_dir}/labels/holdout.txt"

cd "${paddleocr_dir}"
"${python_bin}" tools/train.py -c "${config_path}" -o \
  Global.use_gpu=False \
  Global.epoch_num=10 \
  Global.save_model_dir="${output_dir}" \
  Global.save_epoch_step=1 \
  Global.eval_batch_step='[0, 1]' \
  Global.pretrained_model="${work_dir}/pretrained/PP-OCRv6_small_rec_pretrained" \
  Global.character_dict_path="${paddleocr_dir}/ppocr/utils/dict/ppocrv6_dict.txt" \
  Train.dataset.data_dir="${labels_dir}" \
  Train.dataset.label_file_list="[${labels_dir}/labels/train.txt]" \
  Train.sampler.first_bs=8 \
  Train.loader.batch_size_per_card=8 \
  Train.loader.drop_last=False \
  Train.loader.num_workers=0 \
  Eval.dataset.data_dir="${labels_dir}" \
  Eval.dataset.label_file_list="[${labels_dir}/labels/holdout.txt]" \
  Eval.loader.batch_size_per_card=8 \
  Eval.loader.num_workers=0

evaluation_dir="${work_dir}/evaluations/rec_pp_ocrv6_small/$(date -u +%Y.%m.%d-%H%M%S)-$$"
cd "${root_dir}"
"${root_dir}/training/evaluate_rec_checkpoint.sh" \
  "${output_dir}/best_accuracy" \
  "${evaluation_dir}"
