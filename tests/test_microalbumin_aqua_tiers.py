import importlib.util
import sys
import types
from pathlib import Path

import pytest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    np = None
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _install_stub(name, module):
    if name not in sys.modules:
        sys.modules[name] = module

_matplotlib = types.ModuleType("matplotlib")
_matplotlib.use = lambda *args, **kwargs: None
_pyplot = types.ModuleType("matplotlib.pyplot")
_pyplot.subplots = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pyplot unavailable in unit test"))
_pyplot.close = lambda *args, **kwargs: None
_pyplot.savefig = lambda *args, **kwargs: None
_pyplot.tight_layout = lambda *args, **kwargs: None
_install_stub("matplotlib", _matplotlib)
_install_stub("matplotlib.pyplot", _pyplot)

_cv2 = types.ModuleType("cv2")
_cv2.COLOR_RGB2HSV = 0
_cv2.INTER_NEAREST = 0
_cv2.erode = lambda mask, kernel, iterations=1: mask
_cv2.resize = lambda arr, size, interpolation=None: arr
_cv2.cvtColor = lambda arr, code: arr
_install_stub("cv2", _cv2)

_torch = types.ModuleType("torch")
class _NoGrad:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False
_torch.no_grad = lambda: _NoGrad()
_install_stub("torch", _torch)

_joblib = types.ModuleType("joblib")
_joblib.load = lambda *args, **kwargs: None
_install_stub("joblib", _joblib)

_pil = types.ModuleType("PIL")
_pil_image = types.ModuleType("PIL.Image")
_pil_image.open = lambda *args, **kwargs: None
_pil.Image = _pil_image
_install_stub("PIL", _pil)
_install_stub("PIL.Image", _pil_image)

_alb = types.ModuleType("albumentations")
_alb.Compose = lambda *args, **kwargs: (lambda image: {"image": image})
_alb.Resize = lambda *args, **kwargs: None
_alb.Normalize = lambda *args, **kwargs: None
_alb_pt = types.ModuleType("albumentations.pytorch")
_alb_pt.ToTensorV2 = lambda *args, **kwargs: None
_install_stub("albumentations", _alb)
_install_stub("albumentations.pytorch", _alb_pt)

_ski = types.ModuleType("skimage")
_ski_color = types.ModuleType("skimage.color")
def _rgb2lab(arr):
    import numpy as _np
    return _np.zeros(arr.shape[:-1] + (3,), dtype=float)
_ski_color.rgb2lab = _rgb2lab
_ski_color.deltaE_ciede2000 = lambda a, b: ((a - b) ** 2).sum(axis=-1) ** 0.5
_ski.color = _ski_color
_install_stub("skimage", _ski)
_install_stub("skimage.color", _ski_color)

if HAVE_NUMPY:
    _spec = importlib.util.spec_from_file_location("app_utils_under_test", ROOT / "app" / "utils.py")
    utils = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(utils)
else:
    utils = None

pytestmark = pytest.mark.skipif(not HAVE_NUMPY, reason="numpy is required for synthetic shade-guard array tests")


def _install_synthetic_guard(monkeypatch, *, aqua_fraction, low_fraction, median_aqua=8.0,
                             median_low=9.0, current_value=200.0, median_l=60.0):
    n = 100
    monkeypatch.setattr(utils, "MICROALBUMIN_LOW_EXACT_REFS_LAB", {3: "low3", 10: "low10", 30: "low30"})
    monkeypatch.setattr(utils, "MICROALBUMIN_AQUA_CONFIRM_REFS_LAB", {80: "aqua80", 150: "aqua150"})
    monkeypatch.setattr(utils, "MICRO_STRONG_AQUA_ALLOWED_REFS_LAB", {250: "aqua250", 400: "aqua400"})
    monkeypatch.setattr(utils, "_nearest_micro_chart_bin_with_margin", lambda lab: (80, 8.0, 20.0, 12.0))
    monkeypatch.setattr(utils, "_nearest_micro_allowed_bin_lab", lambda lab, bins: (150 if 150 in bins else bins[0], 5.0))

    def fake_rgb2lab(arr):
        out = np.zeros(arr.shape[:-1] + (3,), dtype=float)
        out[..., 0] = median_l
        return out

    def fake_deltae(lab_pixels, lab_ref):
        vals = np.full(n, 30.0, dtype=float)
        if str(lab_ref).startswith("aqua"):
            vals[:] = median_aqua
        elif str(lab_ref).startswith("low"):
            low_n = int(round(low_fraction * n))
            vals[:low_n] = median_low
            vals[low_n:] = max(median_low + 0.5, utils.MICRO_LOW_DE_MAX + 1.0)
        aqua_n = int(round(aqua_fraction * n))
        if str(lab_ref).startswith("aqua"):
            vals[:aqua_n] = median_aqua
        elif str(lab_ref).startswith("low"):
            vals[:aqua_n] = max(median_aqua + utils.MICRO_AQUA_ADVANTAGE_MARGIN + 5.0, 20.0)
        return vals

    monkeypatch.setattr(utils, "rgb2lab", fake_rgb2lab)
    monkeypatch.setattr(utils, "_deltae_pixels_to_lab_ref", fake_deltae)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    return utils.microalbumin_shade_sanity_check(img, mask, current_value)

def test_existing_weak_aqua_behavior_becomes_moderate(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.30, low_fraction=0.05, median_aqua=8.0, median_low=9.0)
    assert out["moderate_aqua_present"] is True
    assert out["weak_aqua_present"] is True
    assert out["legacy_weak_aqua_present_alias"] is True
    assert out["action"] == "moderate_aqua_provisional_80_150"
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)


def test_new_weak_aqua_maps_to_low_with_low_evidence(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.10, low_fraction=0.60, median_aqua=10.0, median_low=9.0)
    assert out["weak_aqua_low_compatible"] is True
    assert out["action"] == "weak_aqua_low_compatible_mapped_to_low"
    assert out["corrected_albumin_value"] in [3.0, 10.0, 30.0]
    assert out["report_mode"] == "guarded_exact"


def test_very_low_moderate_aqua_maps_to_low_when_competitive(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.23, low_fraction=0.60, median_aqua=10.0, median_low=9.0)
    assert out["very_low_moderate_aqua"] is True
    assert out["action"] == "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
    assert out["report_mode"] == "guarded_exact"


def test_moderate_aqua_does_not_map_to_low_when_low_suppressed(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.28, low_fraction=0.04, median_aqua=8.0, median_low=9.0)
    assert out["moderate_aqua_present"] is True
    assert out["report_mode"] == "provisional_range"


def test_strong_aqua_remains_existing_strong_branch(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.80, low_fraction=0.0, median_aqua=8.0, median_low=20.0)
    assert out["strong_aqua_confirmed"] is True
    assert out["aqua_evidence_tier"] == "strong"
    assert out["action"].startswith("strong_aqua_below_400_matched_")
    assert out["report_mode"] == "exact"


def test_ood_safety_wins_over_low_compatible_weak_aqua(monkeypatch):
    monkeypatch.setattr(utils, "MICRO_CHARTLIKE_DE_MAX", 5.0)
    monkeypatch.setattr(utils, "MICRO_OOD_PIXEL_FRACTION_MAX", 0.50)
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.10, low_fraction=0.60, median_aqua=10.0, median_low=9.0, median_l=90.0)
    assert out["weak_aqua_low_compatible"] is True
    assert out["overbright_ood_no_chart_support"] is True
    assert out["report_mode"] == "unconfirmed"
    assert out["action"] == "unconfirmed_ood_below_400"


def test_backward_compatible_aqua_fields(monkeypatch):
    out = _install_synthetic_guard(monkeypatch, aqua_fraction=0.30, low_fraction=0.05, median_aqua=8.0, median_low=9.0)
    assert "weak_aqua_present" in out
    assert "legacy_weak_aqua_present_alias" in out
    assert "moderate_aqua_present" in out
    assert out["weak_aqua_present"] == out["moderate_aqua_present"]
