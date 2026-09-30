# Platform screenshot-set contract (import side)

This document is the OCRKit-side contract for the platform screenshot-set
endpoint (`OWBastion/owbastion.com#255`). The importer in
`training/importer/` is a read-only client of this contract; it never writes to
the platform and never accesses platform databases or buckets.

A finalized screenshot set supplies immutable member screenshots plus the
provenance a Studio batch needs. It carries **no** annotations, labels, or
split assignments — Studio human review is the training truth.

## Endpoint

```text
GET {base}/v1/ocrkit/screenshot-sets/{version}
```

- `version` is an integer `>= 1` identifying the finalized set revision.
- Authentication is a bearer token passed by OCRKit through
  `OCRKIT_SCREENSHOT_SET_TOKEN` and sent as `Authorization: Bearer <token>`.
- Responses are `Cache-Control: private, no-store` JSON.

| Status | Code | Meaning |
| --- | --- | --- |
| 401 | `UNAUTHENTICATED` | Missing/rejected credentials |
| 404 | `SCREENSHOT_SET_NOT_FOUND` | No set at that version |
| 409 | `SCREENSHOT_SET_NOT_FINALIZED` | Set exists but is not finalized |
| 422 | `INVALID_SCREENSHOT_SET_VERSION` | Non-integer or `< 1` version |

## Set metadata

```json
{
  "schema_version": 1,
  "set_id": "…uuid…",
  "version": 3,
  "finalized": true,
  "finalized_at": "2026-08-01T00:00:00.000Z",
  "members": [
    {
      "source_id": "src-0001",
      "object_key": "evidence/screens/….png",
      "sha256": "…64-hex…",
      "mime_type": "image/png",
      "size_bytes": 12345,
      "layout_version": "1280x720-v6",
      "accuracy": "inaccurate"
    }
  ]
}
```

- `schema_version` must be `1`.
- `finalized` must be `true`; the endpoint rejects unfinalized sets with 409,
  and the importer re-checks the flag anyway.
- `members` is ordered and non-empty; `source_id` and `object_key` are unique
  within the set.
- `sha256` is the member object's hex digest; `size_bytes` is its byte size.
- `layout_version` declares the ROI layout the platform recorded for the
  screenshot; the importer re-detects the decoded image's layout and fails the
  whole import on a mismatch.
- `accuracy` is `"accurate" | "inaccurate" | null` — a review-prioritization
  mark only. It is carried into Studio `cases.json`/review rows as
  `accuracy_feedback` and **never** becomes a transcription or training label.

Every model forbids extra fields. This is the privacy boundary: QQ identity,
player-account internals, Grant/mastery state, risk signals, submission
decisions, evidence URLs, and unrelated platform metadata cannot be smuggled
into an imported batch or its logs.

## Member evidence access

Member bytes are **not** served by this endpoint. The caller downloads each
`object_key` from R2 with a read-only, prefix-scoped Studio key and verifies:

1. `size_bytes` and `sha256` against the set metadata — any mismatch, missing
   object, or undecodable image fails the whole import;
2. the decoded image's detected layout equals `layout_version`;
3. supported MIME types: `image/png`, `image/jpeg`, `image/webp`.

Verified member files live under `<work-root>/set-workspace/set-<version>/objects/`
so an interrupted import resumes without re-downloading verified members.
