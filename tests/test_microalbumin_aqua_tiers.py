import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if "numpy" not in sys.modules and importlib.util.find_spec("numpy") is None:
    np_stub = types.ModuleType("numpy")
    class _FakeArray(list):
        def reshape(self, *args, **kwargs):
            return self
        def astype(self, *args, **kwargs):
            return self
        def tolist(self):
            return list(self)
    def _flatten(value):
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(_flatten(item))
            return out
        return [value]
    def _array(value, dtype=None):
        return _FakeArray(_flatten(value))
    np_stub.array = _array
    np_stub.asarray = _array
    np_stub.uint8 = _array
    np_stub.float64 = float
    np_stub.float32 = float
    np_stub.clip = lambda x, *args, **kwargs: x
    np_stub.ones = lambda *args, **kwargs: _FakeArray([])
    np_stub.zeros = lambda *args, **kwargs: _FakeArray([])
    np_stub.sqrt = lambda x: x ** 0.5
    np_stub.inf = float("inf")
    np_stub.isscalar = lambda obj: isinstance(obj, (int, float, str, bool, type(None)))
    np_stub.ndarray = _FakeArray
    np_stub.bool_ = bool
    sys.modules["numpy"] = np_stub


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

if "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.erode = lambda mask_u8, kernel, iterations=1: mask_u8
    sys.modules["cv2"] = cv2_stub

if "skimage.color" not in sys.modules:
    skimage_stub = types.ModuleType("skimage")
    color_stub = types.ModuleType("skimage.color")
    color_stub.rgb2lab = lambda arr: arr
    color_stub.deltaE_ciede2000 = lambda a, b: 0
    skimage_stub.color = color_stub
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.color"] = color_stub


for _name in ("torch", "joblib"):
    if _name not in sys.modules and importlib.util.find_spec(_name) is None:
        _stub = types.ModuleType(_name)
        if _name == "joblib":
            _stub.load = lambda *args, **kwargs: None
        sys.modules[_name] = _stub

if "albumentations" not in sys.modules:
    alb_stub = types.ModuleType("albumentations")
    alb_stub.Compose = lambda items: (lambda **kwargs: kwargs)
    alb_stub.Resize = lambda *args, **kwargs: None
    alb_stub.Normalize = lambda *args, **kwargs: None
    sys.modules["albumentations"] = alb_stub
if "albumentations.pytorch" not in sys.modules:
    alb_pt_stub = types.ModuleType("albumentations.pytorch")
    alb_pt_stub.ToTensorV2 = lambda *args, **kwargs: None
    sys.modules["albumentations.pytorch"] = alb_pt_stub

spec = importlib.util.spec_from_file_location("utils_aqua_tiers_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def _decision(**evidence):
    tiers = utils.evaluate_microalbumin_aqua_tiers(
        evidence["aqua_pixel_fraction"],
        evidence["median_aqua_de"],
        evidence["low_pixel_fraction"],
        evidence["median_low_de"],
        evidence.get("strong_aqua_confirmed", False),
    )
    decision = utils.microalbumin_guard_from_evidence(
        current_value_float=evidence.get("current_value_float", 150.0),
        low_candidate_class_mg_l=evidence.get("low_candidate_class_mg_l", 10),
        low_shade_confirmed=evidence.get("low_shade_confirmed", False),
        low_shade_confirmed_relaxed=evidence.get("low_shade_confirmed_relaxed", False),
        strong_aqua_confirmed=evidence.get("strong_aqua_confirmed", False),
        high_value_color_verified=evidence.get("high_value_color_verified", False),
        overbright_ood_no_chart_support=evidence.get("overbright_ood_no_chart_support", False),
        weak_aqua_low_compatible=tiers["weak_aqua_low_compatible"],
        very_low_moderate_aqua=tiers["very_low_moderate_aqua"],
        moderate_aqua_present=tiers["moderate_aqua_present"],
    )
    return {**tiers, **decision}


def test_existing_weak_aqua_behavior_becomes_moderate_aqua_provisional():
    out = _decision(
        aqua_pixel_fraction=utils.MICRO_MODERATE_AQUA_MIN,
        median_aqua_de=utils.MICRO_MODERATE_AQUA_DE_MAX,
        low_pixel_fraction=utils.MICRO_MODERATE_AQUA_LOW_SUPPRESSION_MAX,
        median_low_de=20.0,
        strong_aqua_confirmed=False,
    )
    assert out["moderate_aqua_present"] is True
    assert out["weak_aqua_present"] is True
    assert out["legacy_weak_aqua_present_alias"] is True
    assert out["action"] == "moderate_aqua_provisional_80_150"
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)


def test_new_weak_aqua_maps_to_low_when_low_evidence_present():
    out = _decision(
        aqua_pixel_fraction=0.10,
        median_aqua_de=12.0,
        low_pixel_fraction=0.10,
        median_low_de=13.0,
        low_shade_confirmed_relaxed=True,
        strong_aqua_confirmed=False,
        low_candidate_class_mg_l=30,
    )
    assert out["weak_aqua_low_compatible"] is True
    assert out["moderate_aqua_present"] is False
    assert out["action"] == "weak_aqua_low_compatible_mapped_to_low"
    assert out["corrected_albumin_value"] in [3.0, 10.0, 30.0]
    assert out["report_mode"] == "guarded_exact"


def test_very_low_moderate_aqua_maps_to_low_when_low_competitive():
    out = _decision(
        aqua_pixel_fraction=0.23,
        median_aqua_de=12.0,
        low_pixel_fraction=0.10,
        median_low_de=13.0,
        low_shade_confirmed_relaxed=True,
        strong_aqua_confirmed=False,
        low_candidate_class_mg_l=10,
    )
    assert out["very_low_moderate_aqua"] is True
    assert out["action"] == "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
    assert out["report_mode"] == "guarded_exact"


def test_moderate_aqua_does_not_map_to_low_when_low_suppressed():
    out = _decision(
        aqua_pixel_fraction=0.25,
        median_aqua_de=13.0,
        low_pixel_fraction=0.06,
        median_low_de=14.0,
        low_shade_confirmed_relaxed=False,
        strong_aqua_confirmed=False,
    )
    assert out["moderate_aqua_present"] is True
    assert out["report_mode"] == "provisional_range"


def test_strong_aqua_tier_remains_strong_and_not_moderate_override():
    out = _decision(
        aqua_pixel_fraction=0.40,
        median_aqua_de=8.0,
        low_pixel_fraction=0.0,
        median_low_de=15.0,
        strong_aqua_confirmed=True,
    )
    assert out["aqua_evidence_tier"] == "strong"
    assert out["strong_aqua_confirmed"] is True
    assert out["action"] == "strong_aqua_preserved_for_full_guard_matching"


def test_ood_safety_wins_over_weak_low_compatible_aqua():
    out = _decision(
        aqua_pixel_fraction=0.10,
        median_aqua_de=12.0,
        low_pixel_fraction=0.10,
        median_low_de=13.0,
        low_shade_confirmed_relaxed=True,
        overbright_ood_no_chart_support=True,
        strong_aqua_confirmed=False,
    )
    assert out["weak_aqua_low_compatible"] is True
    assert out["report_mode"] == "unconfirmed"
    assert out["action"] == "unconfirmed_ood_below_400"


def test_backward_compatible_aqua_fields_are_present_and_aliased():
    out = _decision(
        aqua_pixel_fraction=0.25,
        median_aqua_de=12.0,
        low_pixel_fraction=0.01,
        median_low_de=20.0,
        strong_aqua_confirmed=False,
    )
    assert "weak_aqua_present" in out
    assert "legacy_weak_aqua_present_alias" in out
    assert "moderate_aqua_present" in out
    assert out["weak_aqua_present"] == out["moderate_aqua_present"]
