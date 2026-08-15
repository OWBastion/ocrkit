"""Offline platform reviewed-snapshot import (issue #5).

The importer consumes one finalized platform-reviewed dataset snapshot and
materializes the private source evidence, crops, and reviewed transcriptions
needed by OCRKit's existing preparation/training workflow. It is offline-only,
read-only against the remote snapshot, and never touches platform databases,
buckets, or business state.
"""

from .client import (
    HttpSnapshotClient,
    ObjectUnavailableError,
    SnapshotAuthError,
    SnapshotClient,
    SnapshotContractError,
    SnapshotNotFinalizedError,
    SnapshotNotFoundError,
)
from .contract import (
    AnnotationsPayload,
    ReviewedAnnotation,
    SnapshotMetadata,
    SnapshotObject,
)
from .importer import (
    ImportReport,
    MissingSourceError,
    SnapshotIntegrityError,
    default_layout_configs,
    import_snapshot,
)
from .split import (
    SPLIT_RULE_VERSION,
    source_split,
    split_rule_parameters,
    split_sources,
)

__all__ = [
    "SPLIT_RULE_VERSION",
    "AnnotationsPayload",
    "HttpSnapshotClient",
    "ImportReport",
    "MissingSourceError",
    "ObjectUnavailableError",
    "ReviewedAnnotation",
    "SnapshotAuthError",
    "SnapshotClient",
    "SnapshotContractError",
    "SnapshotIntegrityError",
    "SnapshotMetadata",
    "SnapshotNotFinalizedError",
    "SnapshotNotFoundError",
    "SnapshotObject",
    "default_layout_configs",
    "import_snapshot",
    "source_split",
    "split_rule_parameters",
    "split_sources",
]
