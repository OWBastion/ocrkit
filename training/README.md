# PP-OCRv6 small training

`training/` is an offline-only workflow. The API image must not install PaddlePaddle, keep training data, or contain training checkpoints.

## Layout

- `datasets/labeled/det/labels.txt`: one line per source image, formatted as `relative/image.png<TAB>[{"transcription":"...","points":[[x1,y1],...,[x4,y4]]}]`.
- `datasets/labeled/rec/labels.txt`: one line per cropped text image, formatted as `relative/crop.png<TAB>transcription`.
- `training/.work/`: local PaddleOCR checkout, environments, checkpoints, exported inference models, and staging artifacts. It is ignored by Git.

Coordinates in detection labels are relative to the ROI image. Recognition labels must contain the exact text expected from the crop. Keep the evaluation screenshots in `datasets/fixtures/challenge`; they are a service regression set, not training data.

## Offline workflow

```bash
./training/bootstrap.sh
uv run python training/scripts/validate_annotations.py det datasets/labeled/det/labels.txt
uv run python training/scripts/validate_annotations.py rec datasets/labeled/rec/labels.txt

# Run the PaddleOCR commands documented in training/configs/*.yaml from the checkout.
# Export det.onnx, rec.onnx and rec_dict.txt to one directory, then copy
# configs/ocr.yaml there as rapidocr.yaml.
uv run python training/scripts/build_manifest.py \
  --artifact-dir training/.work/artifacts/2026-07-11-01 \
  --version 2026-07-11-01
uv run python training/scripts/upload_artifacts.py \
  --artifact-dir training/.work/artifacts/2026-07-11-01 \
  --bucket "$OCRKIT_MODEL_R2_BUCKET"
```

The artifact directory must contain `det.onnx`, `rec.onnx`, `rec_dict.txt`, and `rapidocr.yaml`. `build_manifest.py` writes `manifest.json` with content hashes and versioned Cloudflare R2 object keys. Upload credentials use `OCRKIT_R2_ENDPOINT_URL`, `OCRKIT_R2_ACCESS_KEY_ID`, `OCRKIT_R2_SECRET_ACCESS_KEY`, and optionally `OCRKIT_R2_REGION_NAME` (`auto`).

`upload_artifacts.py` only uploads the files named by the manifest. Publish a manifest under a new version prefix for every release; never overwrite a previously deployed version.
