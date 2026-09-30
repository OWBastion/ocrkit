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

## Model Studio (#6)

Studio is a local-only Svelte/Vite + FastAPI **Model Studio**. Finalized
platform screenshot sets (owbastion.com#255) are the production source intake:
the set supplies immutable member screenshots with checksums and provenance,
and Studio's own human review is the authoritative training truth. Studio owns
only OCRKit model-lifecycle operations. It does not run in the production API
and does not publish a model without an explicit confirmation. The default
launcher starts the API on `127.0.0.1:7860` and the Vite HMR UI on
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

The Model Studio workflow is:

```text
import a finalized platform screenshot set (#26)
→ verified member sources land in a new batch with immutable provenance
→ source-level train/holdout split recorded in batch.json
→ ROI crop → OCR candidates → human review → labels (the training truth)
→ configure/start or continue Smoke training on the reviewed labels
→ evaluate the candidate checkpoint
→ publish an immutable candidate through the existing release gate
→ compare candidate evidence with the current stable manifest
→ explicitly promote to stable, or rollback by selecting an earlier verified manifest
```

`POST /api/screenshot-sets/import` imports one finalized screenshot set by
integer version through the #26 importer (requires
`OCRKIT_SCREENSHOT_SET_BASE_URL` and `OCRKIT_SCREENSHOT_SET_TOKEN`, plus Studio
R2 read access whose `OCRKIT_STUDIO_R2_ALLOWED_PREFIXES` covers the set's
object-key prefix). The same finalized set version cannot be imported twice;
imported batches carry the set identity and per-source provenance in
`batch.json`. Training and publication reuse the same `run_rec_smoke.sh` /
`release_rec_model.sh` scripts and immutable release semantics as the local
workflow.

### Local and R2 imports are supplemental inputs

Platform screenshot sets are the production source intake; Studio's review
workflow turns them into labels. Local file uploads and general R2 imports
remain available for developer fixtures and one-off experiments. They flow
through the same deduplicate → split → crop → review pipeline, carry no
platform provenance, and are never automatically promoted into rules or
production datasets.

### Migration and archival of existing local batches

Existing local batches under `training/.work/studio/batches/<id>/` are
preserved as-is and remain reproducible:

- `batch.json` records the sources, SHA-256 digests, ROI layouts, provenance
  (R2 bucket/object key where applicable), and holdout ratio;
- `dataset/review/*.jsonl` and `labels/*.txt` are the final reviewed state;
- `runs/smoke-*/` and `publication/` retain checkpoints and release logs;
- finalized exports already copied to `datasets/labeled/rec/studio/<id>/` are
  immutable archival packages.

To archive a local batch for historical reproduction, export it through
「标签 → 导出私有数据集」or copy the batch directory to a private location;
the exported package is self-contained (crops, labels, batch manifest,
export.json). Do not silently delete local batches that contain unique private
labels or checkpoints; keep them until an imported screenshot set has been
reviewed to cover the same evidence, and never commit
`training/.work/` or production screenshots to the public repository.

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
Strict-format ROIs are checked before review. For example, `run_code_panel` and
`run_code_right_panel`
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
examples. New batches use the registry to exclude only an identical rejected
crop signature in the same ROI before a row reaches the pending queue. The
rejected text remains provenance for review and is not used as a global
exclusion rule, because the same valid label may appear in another screenshot
or at another position. These negative examples are not written to recognition labels: PP-OCR recognition
training requires a transcription, while the negative registry is the current
candidate-filter path and can later feed a text-detection training workflow.
Overlapping dedicated-field and broad-panel detections are also compared in
normalized screenshot coordinates. When the source, text, and position match,
the dedicated `achievement_panel` or run-code ROI is kept and the duplicate
`left_panel` or `right_panel` row is automatically excluded. A batch may contain
both 16:9 and 16:10 screenshots; Studio records and crops each source with the
matching layout instead of applying one layout to the whole batch.

When ROI coordinates or candidate rules change, use the Studio action
`按最新 ROI 重建候选（保留人工决定）`. It creates a new candidate revision,
matches only unambiguous human decisions by source/ROI/text/position, and sends
unmatched decisions back to review. The previous active dataset is retained under
`dataset-revisions/<revision>/dataset`; training always reads the new active
`dataset` only after review is finalized. The existing Vision and previous-model
refresh actions are reload operations: they update recognition evidence on the
current crops without replacing human decisions.

### Import screenshots from R2

R2 access is used only by the local Studio backend. The browser receives no R2
credentials or object URLs. This import is for a separate non-production
training bucket; do not use the platform evidence bucket. Configure a
read-only key scoped to the training bucket and a narrow prefix allowlist:

```bash
export OCRKIT_R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
export OCRKIT_R2_ACCESS_KEY_ID=<read-only-access-key>
export OCRKIT_R2_SECRET_ACCESS_KEY=<read-only-secret>
export OCRKIT_STUDIO_R2_BUCKET=ocrkit-training-staging
export OCRKIT_STUDIO_R2_ALLOWED_PREFIXES=uploads/
```

Studio lists only the allowed prefixes, accepts supported image types, limits
imports to 200 objects per page and 25 MiB per object by default, deduplicates
by SHA-256, and records the private bucket/key provenance in `batch.json`.
Remote images are copied into the ignored local batch and still require
candidate review. Platform submission screenshots are available to training
only through the finalized screenshot-set importer below, which limits reads
to declared set members.

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

## ROI terminology normalization (#3)

`app/parser/terminology.py` is the versioned, ROI-scoped deterministic
terminology normalization layer shared by production recognition and offline
preparation. Rules live in `configs/terminology.yaml`, keyed by layout version
+ ROI with an allowed canonical term set. Exact aliases and single-character
confusion mappings are adopted only when the result is an allowed term; opt-in
constrained fuzzy matching abstains when candidates tie. Raw OCR evidence is
never overwritten, and holdout truth is never rewritten.

Candidate preparation records per-row terminology evidence; candidate
evaluation reports raw versus post-normalization accuracy and hit / false
correction rates:

```bash
uv run python training/scripts/evaluate_rec_candidates.py
```

## Platform screenshot-set import (#26)

`training/importer/` is the offline importer for one finalized platform
screenshot set. The platform supplies immutable set membership — object key,
SHA-256, size, declared layout version, and an optional accuracy mark —
through the contract documented in `training/importer/CONTRACT.md`, and OCRKit
downloads member screenshots directly from R2 with a read-only,
prefix-scoped key. The set carries no annotations, labels, or split
assignments: imported members become a Studio batch whose human review is the
training truth.

The importer is read-only and needs no platform DB access or broad R2
credentials. It:

1. fetches the finalized set metadata by integer version
   (`GET /v1/ocrkit/screenshot-sets/{version}`);
2. downloads every member object from R2 and verifies `size_bytes` and
   SHA-256 against the platform-signed metadata (missing or corrupt evidence
   fails the import; nothing is substituted or skipped);
3. decodes each image and re-detects its ROI layout, which must match the
   declared `layout_version` so Studio crops with the config the platform
   used;
4. resumes partial imports by reusing already-verified workspace files;
5. returns per-source provenance (set id/version, source id, object key,
   sha256, layout version, accuracy mark) recorded in `batch.json`.

`accuracy` (`"accurate"` | `"inaccurate"` | `null`) is a review-prioritization
hint carried into `cases.json` and review rows as `accuracy_feedback`; it
never becomes a transcription or training label.

Run through Studio (`POST /api/screenshot-sets/import`) or the CLI:

```bash
export OCRKIT_SCREENSHOT_SET_BASE_URL=https://platform.example
export OCRKIT_SCREENSHOT_SET_TOKEN=<token>   # never committed
uv run python training/scripts/import_screenshot_set.py --version 3
```

The workspace under `<work-root>/set-workspace/set-<version>/` caches verified
downloads and is safe to reuse for resume. The Studio route refuses to import
the same finalized set version twice. Imported production evidence is private
and stays out of the public repository, fixture bundle, logs, and released
model artifacts; the metadata contract forbids extra fields so platform
internals cannot leak into a batch.

## Constrained text adjudication experiment (#4)
`training/adjudication/` is an offline, replayable experiment that measures
whether a constrained text-only adjudicator reduces manual review for OCR
terminology cases that deterministic normalization leaves unresolved.

Preconditions before a real go/no-go decision:

1. #3 deterministic normalization is implemented and measured (done);
2. #26 imports a real finalized screenshot set and Studio review produces a
   representative reviewed dataset;
3. the remaining unresolved population is large enough to justify evaluation.

Run the experiment against a reviewed-annotations record file:

```bash
uv run python training/scripts/evaluate_adjudication.py \
  --records training/adjudication/fixtures/reviewed_annotations.jsonl \
  --report training/.work/adjudication/report.json \
  --adjudicator heuristic
```

Records contain only text metadata: schema/layout version, ROI and field
family, per-engine text and confidence, the #3 normalization result, the
allowed canonical candidates, and reviewed ground truth. Extra keys are
forbidden so images, private object URLs, player identity, QQ data, and
submission state cannot enter an experiment or reach a provider.

The heuristic adjudicator is the offline, dependency-free baseline. An
optional OpenAI-compatible provider path is configured through environment
variables (`OCRKIT_ADJUDICATION_ENDPOINT`, `OCRKIT_ADJUDICATION_MODEL`,
`OCRKIT_ADJUDICATION_API_KEY`) and is fail-closed: timeout, invalid output,
off-allowlist candidates, and errors all fall back to `unresolved` for human
review. It never becomes a production OCRKit dependency.

The report compares deterministic-only (arm A) against normalization plus
adjudication (arm B), measures manual-review reduction, precision on resolved
cases, false confident corrections, coverage by field family, cost per 1,000
candidates, and provider failure rate, and applies a maintainer-approved
go/no-go gate. Captured outputs are stored next to the report and can be
replayed later without re-calling the provider:

```bash
uv run python training/scripts/evaluate_adjudication.py \
  --records <records.jsonl> \
  --report training/.work/adjudication/report.json \
  --replay-dir training/.work/adjudication/captured-outputs
```

Do not add a maintained provider adapter unless the experiment on
platform-reviewed annotations shows a material manual-review reduction with
false confident corrections within the gate.

## Jev-Omni pre-review sidecar experiment (#25)

`training/jev/` is an offline, replayable experiment that measures whether the
Jev-Omni typed-decision model can safely reduce human ROI review in Studio.
Per the issue scope it evaluates only the residual rows that the deterministic
auto-accept/auto-reject rules did not decide:

```text
Studio batch review rows (dataset/review/*.jsonl)
→ materialize PreReviewRecord per residual row (crop digest + options + provenance)
→ bounded decision: de-duplicated OCR candidates + "none correct" + "not valid"
→ route: high-confidence candidate → auto-accept, high-confidence not-valid
  → auto-reject, otherwise → human review
→ fit routing threshold on a source-level fit split, report on the held-out
  validation split
```

The experiment is read-only against Studio batches and never feeds Jev output
into labels, training, or production recognition. Jev-derived output lives
only under the report directory and can be deleted or recomputed without
touching the reviewed dataset.

The model runs locally through the community MLX 4-bit conversion
`Ruiruiz30/Jev-Omni-MLX-4bit` (a third-party conversion of a personal project;
keep it removable). It is an optional local dependency: the experiment
spawns a small worker inside a dedicated virtualenv, so nothing here imports
`mlx`/`mlx-vlm` and neither training nor the API gains a dependency:

```bash
mkdir -p training/.work/jev
hf download Ruiruiz30/Jev-Omni-MLX-4bit --local-dir training/.work/jev/Jev-Omni-MLX-4bit
uv venv training/.work/jev/venv --python 3.13
uv pip install --python training/.work/jev/venv/bin/python \
  -r training/.work/jev/Jev-Omni-MLX-4bit/requirements.txt
```

Run the experiment over the local Studio batches:

```bash
uv run python training/scripts/run_jev_experiment.py \
  --runner omni-mlx \
  --model-dir training/.work/jev/Jev-Omni-MLX-4bit \
  --report-dir training/.work/jev/report \
  --records-out training/.work/jev/records.jsonl \
  --image-tokens 35 70 140
```

`--image-tokens` selects the visual-token budget per decision; dense HUD text
may need 70 or higher, so measuring more than one budget is part of the
evaluation. `--max-rows` applies a deterministic stride subsample for smoke
runs. Every decision is captured in `decisions.jsonl` next to the report, so
re-running with `--replay` reproduces the analysis without model calls:

```bash
uv run python training/scripts/run_jev_experiment.py \
  --runner mock --replay training/.work/jev/report/decisions.jsonl \
  --report-dir training/.work/jev/report-check
```

The report records per-budget and per-ROI review reduction, auto-decision
accuracy, false-confident accepts/rejects, the share of rows that always need
manual transcription, threshold curves, post-hoc temperature calibration on
the fit split, latency/memory, and a `keep_sidecar` / `remove` /
`insufficient_data` recommendation against the gate. The lifecycle decision
itself (remove / sidecar / teacher follow-up) remains a maintainer call per
the issue.

### Measured result (local Studio batches, 929 residual rows)

Two prompt variants were evaluated against held-out validation rows at
thresholds fitted on the fit split (gate: ≥25% review reduction, ≤2%
false-confident rate, ≥97% auto-decision accuracy):

| Prompt | Image tokens | Fitted threshold (temp.) | Review reduction | Auto-decision accuracy | False-confident | Auto-rejects |
| --- | --- | --- | --- | --- | --- | --- |
| v1 generic | 35 | 0.765 (T=3.44) | 1.7% | 100% | 0 | 0 |
| v1 generic | 70 | 0.740 (T=3.51) | 9.3% | 100% | 0 | 0 |
| v1 generic | 140 | 0.735 (T=3.70) | 7.3% | 100% | 0 | 0 |
| v2 per-ROI hints | 70 | 0.790 (T=3.89) | 1.7% | 100% | 0 | 0 |

The model's raw confidence is overconfident (ECE ≈ 0.26–0.29); temperature
scaling recovers calibration (ECE ≈ 0.10) but safe thresholds still route
under 10% of rows. Pushing review reduction toward 60% would require
thresholds with ~30% false-confident decisions, which would corrupt training
truth. Auto-reject never engages: on rejected rows the crop usually still
shows text that literally matches an OCR candidate, so "not a valid target
text" fires on ~0.4% of them even with per-ROI content hints — most reject
reasons (duplicate ROI, wrong-source content, mislocalization) are contextual,
not visible in the crop. ~13% of rows additionally need manual transcription
regardless (the accepted text appears in no candidate). Median latency is
~1.6 s/decision at 70 tokens with ~7.4 GB peak Metal memory.

Recommendation: `remove`. Jev's pre-review assistance does not materially
reduce Studio review at any usable safety level; keep the experiment tooling
replayed-from-git but do not retain the model as a standing sidecar.

## Recognition Smoke training

Prepare the offline environment once, then run the CPU recognition Smoke:

```bash
./training/setup_rec_environment.sh
./training/run_rec_smoke.sh
```

`run_rec_smoke.sh` accepts `--labels-dir`, `--output-dir`, `--epochs` (the
target total epoch), and `--resume-checkpoint` (a checkpoint base path without
`.pdparams`, `.pdopt`, or `.states`). `--device cpu|cuda` selects PaddleOCR's
device and defaults to `cpu`; `--train-only` stops after training and pruning
without running the evaluation. It validates both label files, fine-tunes recognition only, and
leaves `latest` plus `best_accuracy` under `training/.work/`. Per-epoch
`iter_epoch_*` dumps and PaddleOCR's duplicate `best_model/` copy are pruned
after training. To reclaim space from older runs:

```bash
uv run python training/scripts/prune_rec_checkpoints.py --root training/.work
```

Training and release use the same evaluator,
`training/evaluate_rec_checkpoint.sh`. For a checkpoint it:

1. downloads and verifies the locked detector;
2. exports the recognition checkpoint to ONNX and copies `rec_dict.txt`;
3. creates a RapidOCR config for the artifact directory; and
4. runs the end-to-end challenge fixture evaluation, writing `fixture_report.json`.

The current release gate is field accuracy at least `364/379`
(`0.9604221635883905`). A failed Smoke keeps the checkpoint and evaluation
report for inspection but returns a non-zero status.

### Run recognition training on Colab GPU

Install the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
and prepare the reviewed/materialized dataset. `setup_rec_environment.sh` is
only needed locally if you also want CPU Smoke or a non-default (custom) base
checkpoint; the default official base checkpoint is fetched by Colab itself.

```bash
uv tool install google-colab-cli
```

The first CLI request can prompt for Google OAuth authentication in the
terminal. The CLI keeps those credentials locally; OCRKit does not send
platform or release credentials to Colab. The runner also requires
`OCRKIT_R2_ENDPOINT_URL`, `OCRKIT_R2_ACCESS_KEY_ID`, `OCRKIT_R2_SECRET_ACCESS_KEY`,
and `OCRKIT_R2_DEFAULT_BUCKET` (see `.env.model.example`): the trained
checkpoint is far too large to transfer efficiently through the Colab CLI, so
Colab uploads it straight to that private R2 bucket using a short-lived,
single-object presigned URL that the runner generates locally and never
writes to disk or a log; Colab never receives R2 credentials. The runner
downloads the checkpoint from R2 and deletes the object once it has done so.

Train on Colab and evaluate the retrieved checkpoint locally with one command:

```bash
uv run python training/run_rec_colab.py
```

The default dataset is `datasets/labeled/rec`. To train from an exported
Studio batch, select the batch's dataset directory explicitly; the runner
reads the sibling `batch.json` for screenshot-set provenance:

```bash
uv run python training/run_rec_colab.py \
  --labels-dir datasets/labeled/rec/studio/<batch-id>/dataset \
  --gpu T4 \
  --epochs 10
```

`--gpu` is a Colab allocation preference (default `T4`), not a model or
training requirement. PaddlePaddle's CUDA runtime and device are checked before
training; an unavailable or unsupported GPU request fails without falling back
to CPU. `--timeout-seconds` (default 6 hours) bounds the remote run; the Colab
CLI's own `exec` default of 30 seconds is always overridden. The local CPU command remains `./training/run_rec_smoke.sh`.

The 2.9 GB `paddlepaddle-gpu` wheel is slow to fetch from the official
CDN outside China, so the runner installs a checksummed mirror of the official
cu129 build (`PADDLE_WHEEL` in `run_rec_colab.py`) when the runtime selects the
cu129 index, and otherwise falls back to the official index.

Only training runs on Colab. CUDA builds of PaddlePaddle export `nn.Linear` as
`linear_v2`, which `paddle2onnx` cannot convert, so the runner retrieves the
checkpoint, stops the runtime, and then runs the unchanged
`training/evaluate_rec_checkpoint.sh` on your machine (the local training
environment from `setup_rec_environment.sh` is required). The run only
succeeds if that evaluation passes the same gate as a local run.

Through the Colab CLI, the runner transfers only the selected train/holdout
labels and referenced crops, available review provenance files, the
training scripts, and (when `--pretrained-checkpoint` names a checkpoint other
than the official default) that custom checkpoint. The official default base
checkpoint is instead fetched by Colab directly from its public URL and
checksum-verified there, and the trained checkpoint returns through R2 rather
than the CLI. It records source revisions, input checksums, the effective
training configuration, PaddleOCR revision, allocated GPU details, checkpoint
checksums, and the local evaluation summary in the returned `run.json`.

The checkpoint, `fixture_report.json`, provenance, the remote training log,
the Colab CLI log, and `status.json` are stored below the ignored
`training/.work/colab-runs/<run-id>/` directory. A successful run does not
publish a candidate or change the stable model channel. Provisioning,
staging, training, metadata retrieval, and handled failures all stop the
Colab runtime after it has been allocated; it is stopped before the local
evaluation starts. Failed runs keep diagnostics and any partial output under
`partial/`; if teardown itself fails, `status.json` includes the named
`colab stop` command to release that session. The R2 checkpoint object is
deleted once retrieved, on both success and failure.

To evaluate a checkpoint explicitly, use a new output directory:

```bash
./training/evaluate_rec_checkpoint.sh \
  training/.work/checkpoints/rec_pp_ocrv6_small/best_accuracy \
  training/.work/evaluations/manual-check
```

### Run recognition training on Kaggle GPU

Kaggle is a second, optional remote GPU backend. It runs the same CUDA
recognition training path and the same local evaluation gate as Colab; only
provisioning, submission, and status/output retrieval differ. Choose whichever
backend has GPU quota available; neither backend changes the model, dataset,
or evaluation contract.

Install the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and authenticate
it. The current CLI's `kaggle auth login` opens an OAuth flow in the browser
and stores the session in `~/.kaggle/credentials.json`; the legacy
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables or a `~/.kaggle/kaggle.json`
API key downloaded from your Kaggle account settings also work. The runner
reads whichever one authenticated the `kaggle` CLI to name the private kernel
after your own account:

```bash
uv tool install kaggle
kaggle auth login
```

Kaggle also requires the account itself to be **phone-verified** before any
kernel can have internet access at all, regardless of `enable_internet` in
the pushed kernel metadata; without it, a kernel's network requests fail with
a DNS resolution error. Verify once at
[kaggle.com](https://www.kaggle.com) → account Settings → Phone Verification
(or from a Notebook's Settings → Internet toggle, which links to the same
verification flow) before running this backend.

The runner also requires `OCRKIT_R2_ENDPOINT_URL`, `OCRKIT_R2_ACCESS_KEY_ID`,
`OCRKIT_R2_SECRET_ACCESS_KEY`, and `OCRKIT_R2_DEFAULT_BUCKET` (see
`.env.model.example`): both the input archive and the trained checkpoint
travel through short-lived, single-object presigned R2 URLs that the runner
generates locally and never writes to disk or a log. Kaggle never receives R2
credentials, platform credentials, or release credentials; the pushed kernel
metadata enables internet access only so it can GET the input archive and PUT
the checkpoint through those two presigned URLs.

Train on Kaggle and evaluate the retrieved checkpoint locally with one command:

```bash
uv run python training/run_rec_kaggle.py
```

```bash
uv run python training/run_rec_kaggle.py \
  --labels-dir datasets/labeled/rec/studio/<batch-id>/dataset \
  --epochs 10
```

Kaggle does not let you pick a specific GPU model (unlike Colab's `--gpu`
preference); the runner always requests the generic `gpu` accelerator, and an
unavailable/exhausted accelerator fails the run explicitly rather than
retrying on CPU. `--timeout-seconds` (default 6 hours) bounds how long the
runner polls kernel status; Kaggle kernel execution is asynchronous, unlike
Colab's synchronous `exec`, so the runner polls `kaggle kernels status`
instead of streaming output live.

The runner builds the identical input archive/`request.json` contract Colab
uploads: the same selected train/holdout labels and referenced crops,
available review provenance files, the training scripts, and (when
`--pretrained-checkpoint` names a checkpoint other than the official default)
that custom checkpoint. Unlike Colab's chunked CLI upload, Kaggle's own
dataset-attachment mechanism is not used for this at all: it silently
auto-extracts or drops archives and subdirectories depending on undocumented,
unstable per-format behavior with no server-side signal to control it (a
`kaggle kernels push` also reads only the pushed script's own text as the
kernel source and ignores every other file in the push folder, so there is no
sibling-file channel either). Instead the archive travels through R2, exactly
like the checkpoint travels back: the runner uploads it to a run-scoped key,
generates a bounded presigned GET URL, and substitutes that URL into
`training/kaggle_remote.py`'s own source text before pushing it as a
**private script kernel** (`kaggle kernels push`) — the only channel available
to hand a Kaggle script kernel any per-run data. The kernel downloads and
verifies the archive, fetches the official base checkpoint or verifies the
uploaded one, runs the unchanged CUDA training/evaluation path, and uploads
the resulting checkpoint to R2.

Because Kaggle committed kernel execution has no interactive runtime to stop,
teardown does not imitate `colab stop`. Instead, once the run finishes (or
fails), the runner always deletes the uploaded input archive from R2, since
it is the only place reviewed training crops are ever staged for Kaggle; any
deletion failure is logged as a warning rather than failing the run (matching
how the retrieved checkpoint object is already cleaned up). The Kaggle CLI has
no kernel-delete command, so the private kernel and its output remain in your
own Kaggle account; `status.json` records `kaggle_kernel` so you can remove it
from [kaggle.com/code](https://www.kaggle.com/code) if you do not want to keep
it. A failed run never publishes a candidate or updates the stable channel.

The checkpoint, `fixture_report.json`, provenance, the remote training log,
the Kaggle CLI log, and `status.json` are stored below the ignored
`training/.work/kaggle-runs/<run-id>/` directory, mirroring the Colab layout.
Failed runs keep diagnostics and any partial output under `partial/`.

`training/remote_gpu_common.py` holds the local-side contract shared by both
backends (input staging, the presigned-URL R2 transfer, checkpoint
retrieval/verification, and the local evaluation gate); `training/colab_remote.py`
and `training/kaggle_remote.py` each stay a single self-contained script
because both Colab's `exec -f` and a Kaggle script kernel's `code_file` only
ever transfer that one file's own content to the remote runtime.

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
  --holdout-labels training/.work/studio/batches/<batch-id>/dataset/labels/holdout.txt \
  --holdout-images-root training/.work/studio/batches/<batch-id>/dataset \
  --provenance training/.work/studio/batches/<batch-id>/batch.json \
  --release-channel models/pp-ocrv6-small/channels/candidate.json
```

The release command generates an unused UTC version, runs the shared fixture
gate and `uv run pytest -q`, records release evidence, builds a content-hashed
manifest, refuses existing objects, uploads an immutable version under
`models/pp-ocrv6-small/<version>/`, downloads and verifies the publication with
RapidOCR, then updates only
`models/pp-ocrv6-small/channels/candidate.json`. It evaluates the isolated
holdout crops at the same `364/379` gate, records the holdout result and
provenance in the immutable manifest, and refuses a direct stable
channel write. The Studio or the explicit commands below compare the candidate
with stable before promotion; missing or failing evidence keeps promotion
closed.

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
uv run python training/scripts/compare_model_channels.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET" \
  --report training/.work/model-comparison.json
uv run python training/scripts/promote_model_channel.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET"
uv run python training/scripts/rollback_model_channel.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET" \
  --manifest-key models/pp-ocrv6-small/<previous-version>/manifest.json
```

`build_manifest.py` fixes the model namespace and records SHA-256 and size for
all four files, plus the release evidence supplied by the candidate workflow.
Uploads are immutable and must use a new version. Candidate publication is
separate from stable selection; promotion records stable-channel history so
rollback can select a previously verified manifest without retraining.

```bash
uv run python training/scripts/publish_model_channel.py \
  --bucket "$OCRKIT_R2_DEFAULT_BUCKET" \
  --channel-key models/pp-ocrv6-small/channels/candidate.json \
  --manifest-key models/pp-ocrv6-small/<version>/manifest.json
```

## Verification and CI

Run the proportionate local checks before sharing a change:

```bash
uv run pytest -q
uv run python scripts/batch_eval.py --min-field-accuracy 0.9604221635883905
cargo test --manifest-path rust/Cargo.toml --workspace --locked
```

The Python pull-request workflow runs tests and the public run-code smoke
fixtures without checking out private datasets. The manual/nightly
`Compatibility` workflow pins the private datasets revision and runs the full
Bastion screenshot gate, retaining its report and log as an artifact. The model
release script runs the full gate again before publishing a candidate. The Rust workflow owns the image
CLI tests and lint. The Docker GHCR workflow intentionally ignores
`training/**`, `scripts/**`, `tests/**`, `rust/**`, and `datasets/**` changes;
training and model publication remain separate from the production image
build.
