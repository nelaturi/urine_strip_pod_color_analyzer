import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if _HAS_NUMPY:
    import numpy as np
else:
    np = None

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy is not installed in this environment")

ROOT = Path(__file__).resolve().parents[1]

# Keep these legacy-recovery tests independent of Flask/TorchScript/joblib loading.
for _name in ("torch", "joblib"):
    if _name not in sys.modules and importlib.util.find_spec(_name) is None:
        _stub = types.ModuleType(_name)
        if _name == "joblib":
            _stub.load = lambda *args, **kwargs: None
        if _name == "torch":
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): return False
            _stub.no_grad = lambda: _NoGrad()
        sys.modules[_name] = _stub

if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
    mpl_stub = types.ModuleType("matplotlib")
    mpl_stub.use = lambda *args, **kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.subplots = lambda *args, **kwargs: (None, [])
    pyplot_stub.close = lambda *args, **kwargs: None
    pyplot_stub.tight_layout = lambda *args, **kwargs: None
    pyplot_stub.savefig = lambda *args, **kwargs: None
    sys.modules["matplotlib"] = mpl_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "PIL" not in sys.modules and importlib.util.find_spec("PIL") is None:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

if "albumentations" not in sys.modules:
    alb_stub = types.ModuleType("albumentations")
    class _Compose:
        def __init__(self, _items):
            self.items = _items
        def __call__(self, image):
            return {"image": image}
    alb_stub.Compose = _Compose
    alb_stub.Resize = lambda *args, **kwargs: ("Resize", args, kwargs)
    alb_stub.Normalize = lambda *args, **kwargs: ("Normalize", args, kwargs)
    sys.modules["albumentations"] = alb_stub

if "albumentations.pytorch" not in sys.modules:
    alb_pt_stub = types.ModuleType("albumentations.pytorch")
    alb_pt_stub.ToTensorV2 = lambda *args, **kwargs: ("ToTensorV2", args, kwargs)
    sys.modules["albumentations.pytorch"] = alb_pt_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.erode = lambda mask_u8, kernel, iterations=1: (mask_u8 > 0).astype(np.uint8)
    cv2_stub.cvtColor = lambda arr, code: np.zeros(arr.shape, dtype=np.uint8)
    cv2_stub.COLOR_RGB2HSV = 0
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 0
    cv2_stub.INTER_NEAREST = 0
    sys.modules["cv2"] = cv2_stub

if "skimage.color" not in sys.modules:
    skimage_stub = types.ModuleType("skimage")
    color_stub = types.ModuleType("skimage.color")
    def _rgb2lab(arr):
        arr = np.asarray(arr, dtype=np.float64)
        out = np.zeros(arr.shape, dtype=np.float64)
        out[..., 0] = arr[..., 0] / 255.0 * 100.0
        return out
    def _deltae(a, b):
        return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)
    color_stub.rgb2lab = _rgb2lab
    color_stub.deltaE_ciede2000 = _deltae
    skimage_stub.color = color_stub
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.color"] = color_stub

if _HAS_NUMPY:
    spec = importlib.util.spec_from_file_location("utils_legacy_recovery_under_test", ROOT / "app" / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
else:
    utils = None


def _mask():
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 2:18] = True
    return m


def _img():
    return np.full((20, 20, 3), 170, dtype=np.uint8)


def _guard(mode="unconfirmed"):
    return {
        "report_mode": mode,
        "action": "v4_action",
        "corrected_albumin_value": None if mode == "unconfirmed" else 150,
    }


def _candidate(version, value, nearest_chart_de=5.0):
    return {
        "version": version,
        "median_rgb": (170, 180, 175),
        "median_lab": (72.0, -6.0, 4.0),
        "continuous_albumin_value": value,
        "mapped_albumin_bin": value,
        "nearest_chart_bin": None if value is None else int(value),
        "nearest_chart_de": nearest_chart_de,
        "low_end_snap_applied": False,
        "reason": "test_candidate",
    }


def _patch_candidates(monkeypatch, v2, v3, de2=5.0, de3=5.0):
    monkeypatch.setattr(utils, "estimate_microalbumin_v2_style", lambda *a, **k: _candidate("v2_style", v2, de2))
    monkeypatch.setattr(utils, "estimate_microalbumin_v3_style", lambda *a, **k: _candidate("v3_style", v3, de3))


def _recover():
    return utils.recover_microalbumin_from_legacy_when_unconfirmed(
        _img(), _img(), _mask(), 100, _guard("unconfirmed"), regression_model=None, allow_liberal=True
    )


def test_no_recovery_when_v4_is_exact():
    out = utils.recover_microalbumin_from_legacy_when_unconfirmed(
        _img(), _img(), _mask(), 100, _guard("exact"), regression_model=None, allow_liberal=True
    )
    assert out["accepted"] is False
    assert out["triggered_by"] == "not_v4_unconfirmed"
    assert out["final_report_mode"] == "unconfirmed"


def test_recovery_triggers_when_v4_is_unconfirmed(monkeypatch):
    _patch_candidates(monkeypatch, 150, 150)
    out = _recover()
    assert out["triggered_by"] == "v4_unconfirmed"
    assert out["accepted"] is True


def test_v2_v3_agreement(monkeypatch):
    _patch_candidates(monkeypatch, 150, 150)
    out = _recover()
    assert out["recovered_albumin_bin"] == 150
    assert out["conflict_status"] == "agreement"


def test_v2_v3_conflict_average_bin(monkeypatch):
    _patch_candidates(monkeypatch, 30, 150)
    out = _recover()
    assert out["mean_candidate_bin"] == 90
    assert out["recovered_albumin_bin"] == 80
    assert out["conflict_status"] == "conflict_average_bin"


def test_higher_uacr_selection(monkeypatch):
    _patch_candidates(monkeypatch, 80, 250)
    out = _recover()
    assert out["recovered_albumin_bin"] == 150
    assert out["uacr_v2"] == 80
    assert out["uacr_v3"] == 250
    assert out["uacr_recovered"] == 150
    assert out["selected_uacr"] == 250
    assert out["selected_uacr_source"] == "v3_style"


def test_single_candidate_available(monkeypatch):
    _patch_candidates(monkeypatch, None, 400)
    out = _recover()
    assert out["recovered_albumin_bin"] == 400
    assert out["conflict_status"] == "single_candidate_available"


def test_no_candidate_available(monkeypatch):
    _patch_candidates(monkeypatch, None, None)
    out = _recover()
    assert out["accepted"] is False
    assert out["final_report_mode"] == "unconfirmed"
    assert out["conflict_status"] == "no_candidate_available"


def test_liberal_mode_accepts_high_nearest_chart_de(monkeypatch):
    _patch_candidates(monkeypatch, 150, 150, de2=30.0, de3=30.0)
    out = _recover()
    assert out["accepted"] is True
    assert out["v2_candidate"]["nearest_chart_de"] == 30.0


def test_display_mode_uses_legacy_recovered_not_exact(monkeypatch):
    _patch_candidates(monkeypatch, 150, 150)
    out = _recover()
    assert out["final_report_mode"] == "legacy_recovered"
    assert out["final_report_mode"] != "exact"


def test_uacr_staging():
    assert utils.stage_uacr_value(29.99)["uacr_stage_code"] == "A1"
    # A1/A2 boundary is inclusive at the A1 side: UACR == 30 -> A1.
    assert utils.stage_uacr_value(30)["uacr_stage_code"] == "A1"
    assert utils.stage_uacr_value(30.01)["uacr_stage_code"] == "A2"
    assert utils.stage_uacr_value(300)["uacr_stage_code"] == "A2"
    assert utils.stage_uacr_value(300.01)["uacr_stage_code"] == "A3"
