# Bastion screenshot compatibility gate

OCRKit records the supported producer revision in
`configs/bastion_screenshot_compatibility.json` without copying Bastion's HUD
contract. The current support boundary is Bastion `settlement-hud-v1`, first
released in `v26.0811.1`. Bastion remains the source of truth for visible HUD
facts; OCRKit owns this consumer matrix and its recognition/evaluation coverage.

Run the gate locally with the checked-in safe run-code fixtures and the private
dataset submodule initialized:

```bash
uv run python scripts/compatibility_gate.py \
  --report training/.work/compatibility-report.json
```

The report retains per-field results and classifies failures as unsupported or
wrong layout selection, ROI/preprocessing quality or rejection, parser or
normalization, or recognition/model accuracy. A failure in any declared
critical field fails the command even when the aggregate field score is high.
The run-code fixture set covers valid, malformed, missing, cropped, ambiguous,
and compressed/scaled evidence. The private released-settlement fixture set
supplies the full current critical-field and 16:10 coverage without putting
player screenshots in this repository.

The same command is intended for model evaluation and candidate promotion. The
full private-corpus gate runs through the manual/nightly `Compatibility` GitHub
Actions workflow and in the model release script; ordinary pull requests run
only the public run-code smoke evaluation so that they do not repeat the full
OCR corpus.
Production rollout must use the promoted immutable manifest and must not treat
this local/CI gate as proof of the platform submission or grant path.
