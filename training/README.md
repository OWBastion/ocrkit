# PP-OCRv6 small training

`training/` is an offline-only workflow. The API image must not install PaddlePaddle, keep training data, or contain training checkpoints.

## OCRKit Studio (local WebUI)

OCRKit Studio is the local recognition-labeling and smoke-training workbench. Its responsive UI
is a Svelte + Vite + Tailwind application served by a local-only FastAPI API; it is not part of
the production OCR service and does not publish models. Imported screenshots, ROI previews,
crops, review manifests, and training logs live only in the ignored `training/.work/studio/` workspace.

```bash
./studio.sh
# Open http://127.0.0.1:5173 (Vite HMR)
```

The default launcher starts the local API with the macOS Vision extra plus the Vite HMR frontend.
Changes under `training/studio/frontend/src/` and `training/studio/` update without a restart. Use
`./studio.sh build` to build only, or `./studio.sh start --port 7861` for the built static preview.

The explicit development form is also available:

```bash
./studio.sh dev
# Open http://127.0.0.1:5173
```

This starts the local Studio API on port `7860` and Vite on port `5173`. Changes under
`training/studio/frontend/src/` update in the browser without restarting either process; `/api`
requests are proxied to the local API. Press `Ctrl+C` to stop both processes.

The WebUI follows one source-safe path:

```text
import screenshots → deterministic source-level train/holdout split → fixed-ROI candidates
→ human review → validated recognition labels → separate CPU smoke process
```

It deduplicates source images by SHA-256, never splits crops from one source screenshot across
train and holdout, and uses atomic replacements when review JSONL files change. The Studio uses
the existing RapidOCR + optional macOS Vision agreement flow; every non-auto-accepted candidate
must be accepted or rejected before labels or training can proceed.

## Layout

- `datasets/labeled/det/labels.txt`: one line per source image, formatted as `relative/image.png<TAB>[{"transcription":"...","points":[[x1,y1],...,[x4,y4]]}]`.
- `datasets/labeled/rec/labels/train.txt` and `labels/holdout.txt`: reviewed cropped text labels, one `relative/crop.png<TAB>transcription` per line.
- `training/.work/`: local PaddleOCR checkout, environments, checkpoints, exported inference models, and staging artifacts. It is ignored by Git.

Coordinates in detection labels are relative to the ROI image. Recognition labels must contain the exact text expected from the crop. Keep the evaluation screenshots in `datasets/fixtures/challenge`; they are a service regression set, not training data.

## Fixture rec smoke run

The fixture preparation command writes only ignored files below `datasets/labeled/rec/`.
It reserves eight whole screenshots for evaluation and creates an editable review manifest for
the remaining screenshots. RapidOCR and Apple Vision both create candidates for each ROI. A
candidate is automatically accepted only when their overlapping text boxes agree after
normalization and both confidence scores are at least 0.98. Review every remaining `pending`
candidate before producing `labels.txt`; do not train against unreviewed OCR output.

```bash
uv sync --extra vision
uv run python training/scripts/prepare_rec_candidates.py
# Review datasets/labeled/rec/review/train.jsonl and review/holdout.jsonl.
# Set every pending row to accepted or rejected and fill transcription for accepted rows.
uv run python training/scripts/finalize_rec_labels.py
uv run python training/scripts/evaluate_rec_candidates.py
./training/bootstrap.sh
./training/setup_rec_environment.sh
./training/run_rec_smoke.sh
```

`run_rec_smoke.sh` performs a ten-epoch CPU fine-tune from the PP-OCRv6 small recognition
checkpoint. It does not train detection and it does not upload any artifact to R2. After training,
it exports `best_accuracy` and runs the same end-to-end fixture gate used by release. The report is
written below `training/.work/evaluations/`; a result below the current `364/379` baseline returns non-zero while keeping
the checkpoint for inspection. In Studio, the training page lists complete checkpoints from prior
Smoke runs. Selecting one creates a new run with PaddleOCR's model, optimizer, and epoch state
restored; set **target total Epoch** higher than the checkpoint's completed epoch to keep training.
Use **加入当前批次** on the import page to add newly collected screenshots to an existing batch;
the original source-level train/holdout assignments stay fixed, while only the added screenshots
receive a split. Generate and review candidates again before finalizing labels. The checkpoint picker
includes complete Smoke checkpoints from every local Studio batch, so a new or expanded batch can
continue from a prior batch's checkpoint.

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
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET"
```

The artifact directory must contain `det.onnx`, `rec.onnx`, `rec_dict.txt`, and `rapidocr.yaml`. `build_manifest.py` writes `manifest.json` with content hashes and versioned Cloudflare R2 object keys. Upload credentials use `OCRKIT_R2_ENDPOINT_URL`, `OCRKIT_R2_ACCESS_KEY_ID`, `OCRKIT_R2_SECRET_ACCESS_KEY`, and optionally `OCRKIT_R2_REGION_NAME` (`auto`).

`upload_artifacts.py` only uploads the files named by the manifest under the reserved
`models/pp-ocrv6-small/` prefix. Publish a manifest under a new version prefix for every release;
never overwrite a previously deployed version. User screenshots belong under `uploads/`.

## Release a trained recognition model

After a manual `run_rec_smoke.sh` run, release the best recognition checkpoint. The command loads
the repository `.env`, downloads and verifies the locked PP-OCRv6 small detector, and generates
an unused UTC version automatically.

```bash
export OCRKIT_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
export OCRKIT_R2_ACCESS_KEY_ID=<r2-access-key-id>
export OCRKIT_R2_SECRET_ACCESS_KEY=<r2-secret-access-key>
export OCRKIT_R2_DEFAULT_BUCKET=ocrkit-models
./training/release_rec_model.sh
```

The command exports `best_accuracy`, runs the full test suite, requires fixture field accuracy
of at least `364/379`, builds a versioned manifest, refuses an already used R2 version, uploads,
then downloads and checksum-validates the published artifact before loading it with RapidOCR.

## Apple Silicon training

On Apple Silicon, the training setup installs the native ARM CPU PaddlePaddle package and ccache
through Homebrew. PP-OCRv6 training remains single-process CPU training on macOS; it does not use
CUDA, Metal, or MPS.
