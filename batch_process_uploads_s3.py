"""S3 batch upload processing helpers.

Mirrors the local batch export fields so S3 output rows can include persisted
segmentation artifact paths.
"""

from batch_process_uploads import SEGMENTATION_ARTIFACT_COLUMNS, _extract_guard_fields

__all__ = ["SEGMENTATION_ARTIFACT_COLUMNS", "_extract_guard_fields"]
