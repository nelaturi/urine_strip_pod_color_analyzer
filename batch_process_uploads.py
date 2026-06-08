"""Batch upload processing helpers.

The full batch runner is not present in this trimmed repository, but these shared
export helpers keep batch CSV schemas aligned with inference payloads.
"""

SEGMENTATION_ARTIFACT_COLUMNS = [
    "segmentation_artifact_dir",
    "segmentation_mask_png",
    "segmentation_overlay_png",
    "pod1_mask_png",
    "pod2_mask_png",
    "pod1_on_strip_png",
    "pod2_on_strip_png",
    "segmentation_metadata_json",
    "pod1_area_px",
    "pod2_area_px",
    "pod1_area_ratio",
    "pod2_area_ratio",
]


def _artifact_file(artifacts, key):
    files = (artifacts or {}).get("artifact_files") or {}
    return files.get(key)


def _extract_guard_fields(result_payload):
    """Extract segmentation artifact fields for appending to batch output rows."""
    artifacts = (result_payload or {}).get("segmentation_artifacts") or {}
    return {
        "segmentation_artifact_dir": artifacts.get("artifact_dir"),
        "segmentation_mask_png": _artifact_file(artifacts, "segmentation_mask_png"),
        "segmentation_overlay_png": _artifact_file(artifacts, "overlay_all_classes_png"),
        "pod1_mask_png": _artifact_file(artifacts, "pod1_mask_png"),
        "pod2_mask_png": _artifact_file(artifacts, "pod2_mask_png"),
        "pod1_on_strip_png": _artifact_file(artifacts, "pod1_on_strip_png"),
        "pod2_on_strip_png": _artifact_file(artifacts, "pod2_on_strip_png"),
        "segmentation_metadata_json": _artifact_file(artifacts, "metadata_json"),
        "pod1_area_px": artifacts.get("pod1_area_px"),
        "pod2_area_px": artifacts.get("pod2_area_px"),
        "pod1_area_ratio": artifacts.get("pod1_area_ratio"),
        "pod2_area_ratio": artifacts.get("pod2_area_ratio"),
    }
