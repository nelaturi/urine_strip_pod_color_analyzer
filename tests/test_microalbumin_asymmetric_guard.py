import importlib.util
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]

# Keep these guard tests independent of the Flask/model stack.
import importlib
for _name in ("torch", "joblib"):
    if _name not in sys.modules and importlib.util.find_spec(_name) is None:
        _stub = types.ModuleType(_name)
        if _name == "joblib":
            _stub.load = lambda *args, **kwargs: None
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

spec = importlib.util.spec_from_file_location("utils_asymmetric_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def _install_guard_diagnostics(
    monkeypatch,
    *,
    median_low_de=20.0,
    low_pixel_fraction=0.0,
    median_aqua_de=20.0,
    aqua_pixel_fraction=0.0,
    median_l=50.0,
    nearest_chart=(None, None, None, None),
):
    n = 100
    monkeypatch.setattr(utils, "MICROALBUMIN_LOW_EXACT_REFS_LAB", {3: "low3", 10: "low10", 30: "low30"})
    monkeypatch.setattr(utils, "MICROALBUMIN_AQUA_CONFIRM_REFS_LAB", {80: "aqua80", 150: "aqua150"})
    monkeypatch.setattr(utils, "MICRO_STRONG_AQUA_ALLOWED_REFS_LAB", {150: "bin150", 250: "bin250", 400: "bin400", 600: "bin600"})
    monkeypatch.setattr(utils, "eroded_mask", lambda mask: mask)
    monkeypatch.setattr(utils, "rgb2lab", lambda arr: np.full(arr.shape, [median_l, 0.0, 0.0], dtype=np.float64))
    monkeypatch.setattr(utils, "_nearest_micro_allowed_bin_lab", lambda lab_obs, allowed_bins: (allowed_bins[0], 0.0))
    monkeypatch.setattr(utils, "_nearest_micro_chart_bin_with_margin", lambda lab_obs: nearest_chart)

    low_hits = int(round(n * low_pixel_fraction))
    aqua_hits = int(round(n * aqua_pixel_fraction))

    def fake_deltae(_lab_pixels, lab_ref):
        vals = np.full(n, 30.0, dtype=np.float64)
        if lab_ref == "low10":
            vals[:] = median_low_de
            vals[:low_hits] = 0.0
        elif lab_ref in {"aqua80", "aqua150", "bin150", "bin250", "bin400", "bin600"}:
            vals[:] = median_aqua_de
            vals[:aqua_hits] = 0.0
        return vals

    monkeypatch.setattr(utils, "_deltae_pixels_to_lab_ref", fake_deltae)


def _run_guard(current_value):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    return utils.microalbumin_shade_sanity_check(img, mask, current_value)


def test_a1_like_weak_aqua_noise_below_threshold_does_not_become_provisional(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=20.0,
        median_aqua_de=12.0,
        aqua_pixel_fraction=0.20,
    )
    out = _run_guard(150)
    assert out["action"] != "weak_aqua_below_400_provisional_80_150"
    assert out["weak_aqua_present"] is False


def test_a1_like_relaxed_low_evidence_maps_to_guarded_exact_low(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=10.0,
        low_pixel_fraction=0.10,
        median_aqua_de=13.0,
    )
    out = _run_guard(150)
    assert out["action"] == "relaxed_low_guard_below_400"
    assert out["report_mode"] == "guarded_exact"
    assert out["corrected_albumin_value"] in {3.0, 10.0, 30.0}


def test_a2_like_aqua_fraction_and_de_triggers_strong_aqua_below_400(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=14.0,
        median_aqua_de=9.0,
        aqua_pixel_fraction=0.46,
    )
    out = _run_guard(250)
    assert out["action"].startswith("strong_aqua_below_400_matched_")
    assert out["corrected_albumin_value"] in [150.0, 250.0, 400.0]


def test_weak_aqua_below_025_does_not_trigger_provisional_range(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=20.0,
        median_aqua_de=10.0,
        aqua_pixel_fraction=0.24,
    )
    out = _run_guard(150)
    assert out["weak_aqua_present"] is False
    assert out["action"] != "weak_aqua_below_400_provisional_80_150"


def test_current_400_with_verified_high_chart_colour_remains_400(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=18.0,
        median_aqua_de=18.0,
        nearest_chart=(400, 4.0, 8.0, 4.0),
    )
    out = _run_guard(400)
    assert out["action"] == "high_value_verified_preserved"
    assert out["corrected_albumin_value"] == 400.0
    assert out["report_mode"] == "exact"


def test_current_600_with_verified_high_colour_is_high_watch_not_low(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=18.0,
        median_aqua_de=18.0,
        nearest_chart=(600, 4.0, 8.0, 4.0),
    )
    out = _run_guard(600)
    assert out["action"] == "high_value_verified_preserved"
    assert out["report_mode"] == "high_watch"
    assert out["corrected_albumin_value"] == 600.0


def test_current_800_with_insufficient_high_evidence_is_unconfirmed_not_10(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=18.0,
        median_aqua_de=18.0,
        nearest_chart=(800, 18.0, 18.5, 0.5),
    )
    out = _run_guard(800)
    assert out["report_mode"] == "unconfirmed"
    assert out["corrected_albumin_value"] is None
    assert out["action"] == "high_value_not_verified_retest"


def test_overbright_ood_high_estimate_is_unconfirmed_not_10(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=22.0,
        median_aqua_de=22.0,
        low_pixel_fraction=0.0,
        aqua_pixel_fraction=0.0,
        median_l=90.0,
    )
    out = _run_guard(600)
    assert out["action"] == "unconfirmed_high_estimate_ood"
    assert out["corrected_albumin_value"] is None
    assert out["corrected_albumin_value"] != 10.0


def test_above_400_values_do_not_pass_through_low_guards(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=10.0,
        low_pixel_fraction=0.10,
        median_aqua_de=13.0,
    )
    out = _run_guard(800)
    assert out["action"] not in {
        "relaxed_low_guard_below_400",
        "guarded_overbright_no_aqua_to_10",
        "confirmed_low_exact_3_10_30",
    }
    assert out["corrected_albumin_value"] != 10.0


def test_batch3_overbright_no_chart_support_regression_is_unconfirmed(monkeypatch):
    _install_guard_diagnostics(
        monkeypatch,
        median_low_de=22.0,
        median_aqua_de=22.0,
        low_pixel_fraction=0.0,
        aqua_pixel_fraction=0.0,
        median_l=90.0,
    )
    out = _run_guard(800)
    assert out["report_mode"] == "unconfirmed"
    assert out["corrected_albumin_value"] is None
    assert out["corrected_albumin_value"] != 10.0
