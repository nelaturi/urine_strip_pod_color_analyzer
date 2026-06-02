import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("utils_low_bin_selector_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def select(medians, fractions=None, creatinine=None, branch="test_branch"):
    return utils.select_low_albumin_bin_3_10_30(
        medians,
        low_pixel_fraction_by_class=fractions,
        creatinine_mg_dl=creatinine,
        branch_name=branch,
    )


def test_clear_3_mg_l_selection():
    out = select({3: 4.0, 10: 5.2, 30: 7.0})
    assert out["selected_low_bin"] == 3.0
    assert out["selection_reason"] == "clear_nearest_low_de"


def test_clear_10_mg_l_selection():
    out = select({3: 6.0, 10: 4.0, 30: 5.3})
    assert out["selected_low_bin"] == 10.0
    assert out["selection_reason"] == "clear_nearest_low_de"


def test_clear_30_mg_l_selection():
    out = select(
        {3: 8.0, 10: 7.0, 30: 4.0},
        {3: 0.01, 10: 0.02, 30: 0.20},
        creatinine=80,
    )
    assert out["selected_low_bin"] == 30.0
    assert out["clear_30_evidence"] is True
    # Diagnostic fields present and well-formed.
    assert out["base_clear_30_evidence"] is True
    assert isinstance(out["low_30_de_advantage"], float)
    assert isinstance(out["low_30_support_dominance"], float)
    assert isinstance(out["boundary_sensitive_30_evidence"], bool)


def test_ambiguous_30_downgraded_to_10_due_to_uacr_boundary_risk():
    # Boundary risk (creat=80 -> UACR@30=37.5). Weak 30 evidence:
    # Adv30 = 5.2-5.0 = 0.2, Dom30 = 0.11-0.10 = 0.01 -> fails confirmation.
    # UACR@10 = 12.5 (<30, A1) and 3 not clearly better than 10 -> select 10.
    out = select(
        {3: 5.4, 10: 5.2, 30: 5.0},
        {3: 0.09, 10: 0.10, 30: 0.11},
        creatinine=80,
    )
    assert out["selected_low_bin"] == 10.0
    assert out["low_30_uacr_boundary_risk"] is True
    assert out["boundary_sensitive_30_evidence"] is False
    assert out["selection_reason"] == "ambiguous_30_downgraded_due_to_uacr_boundary_risk"


def test_boundary_risk_weak_30_downgrades_to_3_when_10_still_a2():
    # creat=20 -> UACR@10 = 50 (>=30, still A2) -> downgrade straight to 3.
    out = select(
        {3: 5.4, 10: 5.2, 30: 5.0},
        {3: 0.09, 10: 0.10, 30: 0.11},
        creatinine=20,
    )
    assert out["selected_low_bin"] == 3.0
    assert out["low_30_uacr_boundary_risk"] is True
    assert out["boundary_sensitive_30_evidence"] is False
    assert out["selection_reason"] == "ambiguous_30_downgraded_due_to_uacr_boundary_risk"


def test_boundary_risk_strong_30_evidence_keeps_30():
    # Boundary risk (creat=80). Strong color advantage:
    # Adv30 = min(8,7)-4 = 3.0 (>=3.0) -> confirmation passes -> keep 30.
    out = select(
        {3: 8.0, 10: 7.0, 30: 4.0},
        {3: 0.01, 10: 0.02, 30: 0.20},
        creatinine=80,
    )
    assert out["selected_low_bin"] == 30.0
    assert out["low_30_uacr_boundary_risk"] is True
    assert out["boundary_sensitive_30_evidence"] is True
    assert out["clear_30_evidence"] is True


def test_non_boundary_risk_30_support_keeps_30():
    # No creatinine -> no boundary risk -> legacy pixel-support path keeps 30
    # even though boundary-sensitive confirmation would fail.
    out = select(
        {3: 6.0, 10: 5.8, 30: 5.2},
        {3: 0.05, 10: 0.06, 30: 0.22},
    )
    assert out["selected_low_bin"] == 30.0
    assert out["low_30_uacr_boundary_risk"] is False
    assert out["base_clear_30_evidence"] is True
    assert out["boundary_sensitive_30_evidence"] is False


def test_ambiguous_low_defaults_to_10():
    out = select({3: 5.1, 10: 5.0, 30: 5.2}, {3: 0.10, 10: 0.11, 30: 0.10})
    assert out["selected_low_bin"] == 10.0
    assert out["selection_reason"] == "ambiguous_default_low_bin"


def test_ambiguous_low_selects_3_when_3_is_clearly_better_than_10():
    out = select({3: 4.7, 10: 5.4, 30: 5.1})
    assert out["selected_low_bin"] == 3.0


def test_pixel_support_tiebreaker_selects_supported_bin():
    out = select({3: 5.0, 10: 5.1, 30: 5.2}, {3: 0.20, 10: 0.11, 30: 0.10})
    assert out["selected_low_bin"] == 3.0
    assert out["selection_reason"] == "pixel_support_tiebreak"


def test_selector_result_is_json_serializable():
    out = select({3: 5.1, 10: 5.0, 30: 5.2}, {"3": 0.10, "10": 0.11, "30": 0.10})
    json.dumps(out)


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
        low_candidate_class_mg_l=evidence.get("low_candidate_class_mg_l", 30),
        low_shade_confirmed=evidence.get("low_shade_confirmed", False),
        low_shade_confirmed_relaxed=evidence.get("low_shade_confirmed_relaxed", False),
        strong_aqua_confirmed=evidence.get("strong_aqua_confirmed", False),
        high_value_color_verified=evidence.get("high_value_color_verified", False),
        overbright_ood_no_chart_support=evidence.get("overbright_ood_no_chart_support", False),
        weak_aqua_low_compatible=tiers["weak_aqua_low_compatible"],
        very_low_moderate_aqua=tiers["very_low_moderate_aqua"],
        moderate_aqua_present=tiers["moderate_aqua_present"],
        median_de_low_by_class=evidence.get("median_de_low_by_class"),
        low_pixel_fraction_by_class=evidence.get("low_pixel_fraction_by_class"),
        creatinine_mg_dl=evidence.get("creatinine_mg_dl"),
    )
    return {**tiers, **decision}


def test_weak_aqua_low_compatible_branch_uses_selector_for_10_not_forced_30():
    out = _decision(
        aqua_pixel_fraction=0.10,
        median_aqua_de=12.0,
        low_pixel_fraction=0.10,
        median_low_de=13.0,
        low_shade_confirmed_relaxed=True,
        low_candidate_class_mg_l=30,
        median_de_low_by_class={3: 5.4, 10: 5.0, 30: 5.2},
        low_pixel_fraction_by_class={3: 0.09, 10: 0.12, 30: 0.10},
        creatinine_mg_dl=80,
    )
    assert out["action"] == "weak_aqua_low_compatible_mapped_to_low"
    assert out["corrected_albumin_value"] == 10.0


def test_very_low_moderate_branch_uses_selector_for_3():
    out = _decision(
        aqua_pixel_fraction=0.23,
        median_aqua_de=12.0,
        low_pixel_fraction=0.10,
        median_low_de=13.0,
        low_shade_confirmed_relaxed=True,
        low_candidate_class_mg_l=30,
        median_de_low_by_class={3: 4.7, 10: 5.4, 30: 5.1},
        creatinine_mg_dl=80,
    )
    assert out["action"] == "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
    assert out["corrected_albumin_value"] == 3.0


def test_moderate_aqua_branch_unchanged():
    out = _decision(
        aqua_pixel_fraction=0.25,
        median_aqua_de=13.0,
        low_pixel_fraction=0.06,
        median_low_de=14.0,
        low_shade_confirmed_relaxed=False,
        strong_aqua_confirmed=False,
        median_de_low_by_class={3: 4.0, 10: 5.0, 30: 5.5},
    )
    assert out["action"] == "moderate_aqua_provisional_80_150"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)
    assert out["low_bin_selection"] is None


def test_strong_aqua_branch_unchanged():
    out = _decision(
        aqua_pixel_fraction=0.40,
        median_aqua_de=8.0,
        low_pixel_fraction=0.0,
        median_low_de=15.0,
        strong_aqua_confirmed=True,
        median_de_low_by_class={3: 4.0, 10: 5.0, 30: 5.5},
    )
    assert out["action"] == "strong_aqua_preserved_for_full_guard_matching"
    assert out["low_bin_selection"] is None
