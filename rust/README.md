# OCRKit image core

This workspace contains the first-stage Rust implementation for deterministic
layout and image preflight work.

The authoritative layout remains
configs/roi_1280x720.yaml. Export its browser/native-safe manifest with:

~~~bash
uv run python scripts/export_layout_manifest.py \
  configs/roi_1280x720.yaml \
  configs/roi_1280x720.manifest.json
~~~

Validate that the checked-in manifest is current:

~~~bash
uv run python scripts/export_layout_manifest.py \
  configs/roi_1280x720.yaml \
  configs/roi_1280x720.manifest.json \
  --check
~~~

The native preflight CLI validates the manifest, evaluates the source aspect
ratio, and maps standard-layout ROI coordinates onto the source image:

~~~bash
cargo run --manifest-path rust/Cargo.toml -p ocrkit-image-cli -- \
  inspect \
  --manifest configs/roi_1280x720.manifest.json \
  --width 2560 \
  --height 1440
~~~

For Studio training-data preparation, the crop-batch command reads a cases
manifest, decodes each source image, resizes it to the standard canvas, writes
raw ROI PNGs, and records source/crop SHA-256 values:

~~~bash
cargo run --manifest-path rust/Cargo.toml -p ocrkit-image-cli -- \
  crop-batch \
  --manifest configs/roi_1280x720.manifest.json \
  --cases training/.work/studio/batches/<batch-id>/cases.json \
  --input-root training/.work/studio/batches/<batch-id> \
  --output-dir /private/tmp/ocrkit-rust-crops
~~~

This phase does not replace Python/OpenCV preprocessing, OCR, parsing, or
platform review logic. The CLI and core are the native test surface for the
future WASM binding.
