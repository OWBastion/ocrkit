# PP-OCRv6 small training and release

`training/` contains the offline OCR model workflow. PaddlePaddle, PaddleOCR,
training data, checkpoints, and exported artifacts must stay out of the API
image. OCRKit publishes recognition evidence; it does not decide whether a
submission is approved or whether a player receives a title.

The current supported model path fine-tunes the PP-OCRv6 small recognition
model. The detector is not trained by the current scripts: it is downloaded
from and verified against `training/configs/pp_ocrv6_small_det.lock.json` for
each evaluation or release. `training/configs/det_pp_ocrv6_small.yaml` is an
upstream detector recipe overlay for reference, not an active end-to-end
detector training command.

## Prerequisites

Initialize the private datasets submodule before running fixture or dataset
commands:

```bash
git submodule update --init --recursive
uv sync --extra dev
```

OCRKit Studio and the candidate workflow use Apple Vision for a second OCR
candidate. On macOS, install the optional dependency with:

```bash
uv sync --extra vision
```

Studio also needs `pnpm`. Its default crop backend invokes the Rust image CLI;
install a Rust toolchain, or set `OCRKIT_RUST_IMAGE_CLI` to a prebuilt
`ocrkit-image-cli` executable. Without that variable, Studio invokes Cargo for
each candidate batch.

`training/setup_rec_environment.sh` creates `training/.work/PaddleOCR`, a
Python 3.12 virtual environment, the CPU PaddlePaddle/PaddleOCR dependencies,
`paddle2onnx`, and the PP-OCRv6 small recognition checkpoint. On Apple Silicon
it installs and checks `ccache`, ARM64 CPU PaddlePaddle, and explicitly avoids
CUDA, Metal, and MPS.

All generated training state belongs below the ignored `training/.work/`
directory. Do not commit production screenshots, debug crops, credentials,
checkpoints, or model binaries.

## OCRKit Studio

Studio is a local-only Svelte/Vite + FastAPI workbench. It does not run in the
production API and does not publish a model without an explicit confirmation.
The default launcher starts the API on `127.0.0.1:7860` and the Vite HMR UI on
`127.0.0.1:5173`:

```bash
./studio.sh
# Open http://127.0.0.1:5173
```

Equivalent commands are:

```bash
./studio.sh dev                 # API + Vite HMR
./studio.sh build               # install locked frontend deps and build only
./studio.sh start --port 7861  # build, then serve the static UI
```

The Studio workflow is:

```text
import local/R2 screenshots
→ SHA-256 deduplicate and split whole source screenshots into train/holdout
→ Rust fixed-ROI crop export with provenance
→ previous OCR artifact + RapidOCR + Apple Vision candidates
→ human review and transcription correction
→ validated labels
→ CPU recognition Smoke
→ optional explicit R2 publication
```

Studio stores batches, source screenshots, crops, review JSONL, logs, and
checkpoints in `training/.work/studio/`. The source-level split is preserved
when screenshots are added to an existing batch; only new source screenshots
receive a split. Re-running candidate generation reuses completed review data,
and **补回 Vision** updates Vision fields without overwriting manual accepted or
rejected decisions.

Rows for which RapidOCR and Vision agree after normalization at confidence at
least `0.98` are automatically accepted, but remain visible and editable.
Strict-format ROIs are checked before review. For example, `run_code_panel`
must contain a valid `本局代码`/`Run Code` value with three four-digit groups;
text from another HUD position is automatically rejected, so it does not enter
the pending queue or become a training label.
When a complete local model artifact exists below `training/.work/artifacts/`,
Studio also loads the newest artifact as a previous-model reference. Train
rows where the previous model and RapidOCR agree at confidence at least `0.98`
are automatically accepted and remain visible for spot checking. Apple Vision
remains visible as a third reference, but does not block this Train decision.
Previous-model suggestions never auto-accept holdout rows.
Every remaining row must be manually accepted with a transcription or rejected
before labels can be generated. Set
`OCRKIT_STUDIO_CANDIDATE_ARTIFACT_DIR` to pin a specific local artifact when
the newest artifact is not the desired previous model.

Manual rejections are also stored locally in
`training/.work/studio/negative-candidates.jsonl` as ROI-scoped negative
examples. New batches use the registry to exclude matching text or crop
signatures across every panel before a row reaches the pending queue. These
negative examples are not written to recognition labels: PP-OCR recognition
training requires a transcription, while the negative registry is the current
candidate-filter path and can later feed a text-detection training workflow.
Overlapping `achievement_panel` and `left_panel` detections are also compared
in normalized screenshot coordinates. When the source, text, and position
match, the more specific `achievement_panel` row is kept and the duplicate
`left_panel` row is automatically excluded.

### Import screenshots from R2

R2 access is used only by the local Studio backend. The browser receives no R2
credentials or object URLs. Configure a read-only key and a narrow allowlist:

```bash
export OCRKIT_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
export OCRKIT_R2_ACCESS_KEY_ID=<read-only-access-key>
export OCRKIT_R2_SECRET_ACCESS_KEY=<read-only-secret>
export OCRKIT_STUDIO_R2_BUCKET=owbastion-codes-evidence
export OCRKIT_STUDIO_R2_ALLOWED_PREFIXES=uploads/
```

Studio lists only the allowed prefixes, accepts supported image types, limits
imports to 200 objects per page and 25 MiB per object by default, deduplicates
by SHA-256, and records the private bucket/key provenance in `batch.json`.
Remote screenshots are copied into the ignored local batch and still require
candidate review.

### Export and continue a batch

**导出到私有 datasets** finalizes and validates the labels, then creates an
immutable package at:

```text
datasets/labeled/rec/studio/<batch-id>/
```

The export contains crops, review manifests, labels, `batch.json`, and
provenance. It refuses to overwrite an existing batch and never runs `git
commit` or `git push`. A complete local Smoke checkpoint from the current or
another Studio batch can be selected as the starting checkpoint for a new
run. Set the target total Epoch above the checkpoint's completed epoch when
continuing training.

## Dataset layout

Recognition labels use one tab-separated sample per line:

```text
relative/crop.png<TAB>exact transcription
```

The normal repository paths are:

```text
datasets/labeled/rec/
├── images/                 # private cropped images
├── review/train.jsonl      # candidate and human-review records
├── review/holdout.jsonl
└── labels/
    ├── train.txt
    └── holdout.txt
```

Detection labels, when used by an offline experiment, contain one source image
per line followed by JSON annotations with four points. Validate either format
with:

```bash
uv run python training/scripts/validate_annotations.py rec datasets/labeled/rec/labels/train.txt
uv run python training/scripts/validate_annotations.py det datasets/labeled/det/labels.txt
```

The screenshots in `datasets/fixtures/challenge` are the service regression
set. They are not automatically training data and must not be replaced by
production evidence without an approved private-dataset change.

`tests/fixtures/run_code` contains synthetic, non-player settlement-layout
fixtures for the run-code field. The standard batch evaluation reports these
separately and requires an exact-match run-code result, including missing,
malformed, ambiguous, compressed, and scaled cases.

## Standalone recognition candidate workflow

This is the script equivalent of the Studio candidate step. It currently
requires macOS and the `vision` extra because it runs both RapidOCR and Apple
Vision:

```bash
uv run python training/scripts/prepare_rec_candidates.py --crop-backend rust
# Review datasets/labeled/rec/review/train.jsonl and review/holdout.jsonl.
# Every row must end with review_status=accepted or review_status=rejected.
uv run python training/scripts/finalize_rec_labels.py
uv run python training/scripts/evaluate_rec_candidates.py
```

Omit `--crop-backend rust` to use the Python crop implementation. The
preparation script writes only below `datasets/labeled/rec/`; review output is
not a substitute for human approval.

## Recognition Smoke training

Prepare the offline environment once, then run the CPU recognition Smoke:

```bash
./training/setup_rec_environment.sh
./training/run_rec_smoke.sh
```

`run_rec_smoke.sh` accepts `--labels-dir`, `--output-dir`, `--epochs` (the
target total epoch), and `--resume-checkpoint` (a checkpoint base path without
`.pdparams`, `.pdopt`, or `.states`). It validates both label files, fine-tunes
recognition only, and leaves checkpoints under `training/.work/`.

Training and release use the same evaluator,
`training/evaluate_rec_checkpoint.sh`. For a checkpoint it:

1. downloads and verifies the locked detector;
2. exports the recognition checkpoint to ONNX and copies `rec_dict.txt`;
3. creates a RapidOCR config for the artifact directory; and
4. runs the end-to-end challenge fixture evaluation, writing `fixture_report.json`.

The current release gate is field accuracy at least `364/379`
(`0.9604221635883905`). A failed Smoke keeps the checkpoint and evaluation
report for inspection but returns a non-zero status.

To evaluate a checkpoint explicitly, use a new output directory:

```bash
./training/evaluate_rec_checkpoint.sh \
  training/.work/checkpoints/rec_pp_ocrv6_small/best_accuracy \
  training/.work/evaluations/manual-check
```

## Release a recognition model

Release requires R2 credentials and a model bucket. The script loads values
from the repository `.env` when present, but credentials must never be
committed:

```bash
export OCRKIT_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
export OCRKIT_R2_ACCESS_KEY_ID=<r2-access-key-id>
export OCRKIT_R2_SECRET_ACCESS_KEY=<r2-secret-access-key>
export OCRKIT_R2_DEFAULT_BUCKET=ocrkit-models

./training/release_rec_model.sh
```

The default checkpoint is
`training/.work/checkpoints/rec_pp_ocrv6_small/best_accuracy`. Studio passes a
batch checkpoint explicitly; this is also available from the command line:

```bash
./training/release_rec_model.sh \
  --checkpoint training/.work/studio/batches/<batch-id>/runs/<run>/checkpoints/best_accuracy \
  --release-channel models/pp-ocrv6-small/channels/stable.json
```

The release command generates an unused UTC version, runs the shared fixture
gate and `uv run pytest -q`, builds a content-hashed manifest, refuses existing
objects, uploads an immutable version under
`models/pp-ocrv6-small/<version>/`, downloads and verifies the publication with
RapidOCR, then updates the requested release channel. A production container
configured with `OCRKIT_MODEL_RELEASE_CHANNEL_KEY` adopts the channel target
after restart or recreation; a release never overwrites an older model
version.

## Manual artifact operations

The release script is the normal path. For an already prepared artifact
directory containing `det.onnx`, `rec.onnx`, `rec_dict.txt`, and `rapidocr.yaml`:

```bash
uv run python training/scripts/build_manifest.py \
  --artifact-dir training/.work/artifacts/<version> \
  --version <version>
uv run python training/scripts/upload_artifacts.py \
  --artifact-dir training/.work/artifacts/<version> \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET"
uv run python training/scripts/verify_published_artifact.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET" \
  --manifest-key models/pp-ocrv6-small/<version>/manifest.json
```

`build_manifest.py` fixes the model namespace and records SHA-256 and size for
all four files. Uploads are immutable and must use a new version. Publish a
channel only after download and RapidOCR verification:

```bash
uv run python training/scripts/publish_model_channel.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET" \
  --channel-key models/pp-ocrv6-small/channels/stable.json \
  --manifest-key models/pp-ocrv6-small/<version>/manifest.json
```

## Verification and CI

Run the proportionate local checks before sharing a change:

```bash
uv run pytest -q
uv run python scripts/batch_eval.py --min-field-accuracy 0.9604221635883905
cargo test --manifest-path rust/Cargo.toml --workspace --locked
```

The Python workflow runs tests and the fixture gate with the private datasets
revision pinned by the repository submodule. The Rust workflow owns the image
CLI tests and lint. The Docker GHCR workflow intentionally ignores
`training/**`, `scripts/**`, `tests/**`, `rust/**`, and `datasets/**` changes;
training and model publication remain separate from the production image
build.
