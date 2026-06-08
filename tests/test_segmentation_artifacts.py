import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _dep in ("numpy", "PIL", "cv2", "torch", "albumentations", "skimage", "joblib"):
    if importlib.util.find_spec(_dep) is None:
        pytest.skip(f"{_dep} is not installed", allow_module_level=True)

import numpy as np

from app.utils import (
    POD1_IDX,
    POD2_IDX,
    create_segmentation_artifact_dir,
    save_segmentation_artifacts,
)


def test_synthetic_segmentation_artifacts_save_expected_files(tmp_path):
    raw = np.zeros((24, 32, 3), dtype=np.uint8)
    raw[..., 0] = 100
    raw[..., 1] = 120
    raw[..., 2] = 140
    wb = raw.copy()
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[3:9, 4:12] = POD1_IDX
    mask[12:20, 16:28] = POD2_IDX
    model_mask = np.zeros((256, 256), dtype=np.uint8)
    artifact_dir = create_segmentation_artifact_dir(tmp_path, request_id="req-1", image_name="strip.png")

    info = save_segmentation_artifacts(
        artifact_dir,
        raw,
        wb,
        mask,
        pred_mask_model_size=model_mask,
        pod1_mask=mask == POD1_IDX,
        pod2_mask=mask == POD2_IDX,
        metadata={"request_id": "req-1", "image_name": "strip.png", "model_path": "model.pt"},
    )

    expected = [
        "raw_image.png",
        "white_balanced_image.png",
        "segmentation_mask_original.npy",
        "segmentation_mask_original.png",
        "segmentation_mask_model_size.npy",
        "pod1_creatinine_mask.npy",
        "pod1_creatinine_mask.png",
        "pod2_microalbumin_mask.npy",
        "pod2_microalbumin_mask.png",
        "pod1_creatinine_on_strip.png",
        "pod2_microalbumin_on_strip.png",
        "segmentation_overlay_all_classes.png",
        "metadata.json",
    ]
    for name in expected:
        assert (Path(artifact_dir) / name).exists(), name

    assert Path(info["artifact_files"]["pod1_on_strip_png"]).exists()
    assert Path(info["artifact_files"]["pod2_on_strip_png"]).exists()
    assert info["database_export_ready"] is True

    metadata = json.loads((Path(artifact_dir) / "metadata.json").read_text())
    json.dumps(metadata)
    assert metadata["database_export_ready"] is True
    assert metadata["pod1_area_px"] == 48
    assert metadata["pod2_area_px"] == 96
    assert metadata["pod1_area_ratio"] == 48 / (24 * 32)
    assert metadata["pod2_area_ratio"] == 96 / (24 * 32)
