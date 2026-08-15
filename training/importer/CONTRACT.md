# Platform reviewed-snapshot contract (import side)

This document is the OCRKit-side contract that the platform snapshot endpoint
(`OWBastion/owbastion.com#105`) implements. The importer in
`training/importer/` is a read-only client of this contract; it never writes to
the remote snapshot and never accesses platform databases or buckets.

## Endpoints

All responses are JSON (`application/json`). Authentication is a bearer token
passed by OCRKit through `OCRKIT_PLATFORM_SNAPSHOT_TOKEN` and sent as
`Authorization: Bearer <token>`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/snapshots/{snapshot_id}` | Immutable finalized snapshot metadata |
| GET | `/api/v1/snapshots/{snapshot_id}/annotations` | Reviewed annotation records |
| GET | `/api/v1/objects/{object_id}/download` | Bounded per-object image download |

Errors: `401/403` (access denied), `404` (missing), other `5xx`/timeouts are
reported as contract failures. Timeouts and concurrency are bounded by the
client.

## Snapshot metadata

```json
{
  "schema_version": 1,
  "snapshot_id": "2026-08-01-final",
  "version": "v3",
  "finalized": true,
  "finalized_at": "2026-08-01T00:00:00Z",
  "objects": [
    {
      "object_id": "obj-0001",
      "kind": "source",
      "sha256": "…",
      "mime_type": "image/png",
      "size_bytes": 12345,
      "source_id": "src-0001",
      "layout_version": "1280x720-v6"
    },
    {
      "object_id": "obj-0002",
      "kind": "crop",
      "sha256": "…",
      "mime_type": "image/png",
      "size_bytes": 456,
      "annotation_id": "ann-0001"
    }
  ]
}
```

- `schema_version` must be `1`.
- `finalized` must be `true`; a non-finalized snapshot is rejected.
- `kind: "source"` objects are source screenshots and must declare
  `source_id` and `layout_version`. `kind: "crop"` objects are platform
  pre-cropped training samples and must declare the owning `annotation_id`.
- Every member object is listed here; the importer materializes only
  snapshot-member evidence.
- Supported image MIME types are `image/png` and `image/jpeg`.
- `sha256` is the hex SHA-256 of the object bytes; the importer verifies every
  download and fails on mismatch.

## Reviewed annotations

```json
{
  "schema_version": 1,
  "snapshot_id": "2026-08-01-final",
  "annotations": [
    {
      "annotation_id": "ann-0001",
      "source_id": "src-0001",
      "layout_version": "1280x720-v6",
      "roi": "left_panel",
      "field": "challenge_stats",
      "ocr_prediction": "编益",
      "exact_transcription": "增益",
      "canonical_value": "增益",
      "box": [[40, 20], [120, 20], [120, 30], [40, 30]]
    }
  ]
}
```

Semantics kept distinct (never conflated):

- `ocr_prediction`: original OCR output (may be null).
- `exact_transcription`: reviewed exact visible text — the only source of OCR
  labels.
- `canonical_value`: business-normalized platform value (may be null).

Each annotation declares exactly one crop source, by priority:

1. `crop_object_id` — a platform pre-cropped training sample (must be a
   `kind: "crop"` object in the snapshot).
2. `box` — a text-line polygon of four `[x, y]` points in **standard-size**
   coordinates of the annotation's layout; OCRKit derives the line crop from
   the normalized source.
3. `roi` — a known field in the annotation's layout; OCRKit derives the ROI
   crop with its own versioned layout/ROI tooling.

`annotation_id` and `source_id` are unique within the snapshot, and every
`source_id` must reference a snapshot member source with the same
`layout_version`.

## Privacy boundary

All payload models reject unknown keys, so QQ identity, player-account
internals, Grant/mastery state, risk signals, submission decisions, image
bytes, and object URLs cannot be passed through the contract into a
materialized import. OCRKit never logs credentials, image bytes, or object
URLs, and imported production evidence stays out of the public repository,
fixture bundles, and released model artifacts.
