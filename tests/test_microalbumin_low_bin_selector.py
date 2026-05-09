import importlib.util
import json
import sys
import types
from pathlib import Path

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


if "skimage.color" not in sys.modules:
    skimage_stub = types.ModuleType("skimage")
    color_stub = types.ModuleType("skimage.color")
    color_stub.rgb2lab = lambda arr: arr
    color_stub.deltaE_ciede2000 = lambda a, b: 0
    skimage_stub.color = color_stub
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.color"] = color_stub

for _name in ("cv2", "torch", "joblib"):
    if _name not in sys.modules and importlib.util.find_spec(_name) is None:
        _stub = types.ModuleType(_name)
        if _name == "joblib":
            _stub.load = lambda *args, **kwargs: None
        sys.modules[_name] = _stub

if "albumentations" not in sys.modules and importlib.util.find_spec("albumentations") is None:
    alb_stub = types.ModuleType("albumentations")
    alb_stub.Compose = lambda items: (lambda **kwargs: kwargs)
    alb_stub.Resize = lambda *args, **kwargs: None
    alb_stub.Normalize = lambda *args, **kwargs: None
    alb_stub.__path__ = []
    sys.modules["albumentations"] = alb_stub
if "albumentations.pytorch" not in sys.modules:
    alb_pt_stub = types.ModuleType("albumentations.pytorch")
    alb_pt_stub.ToTensorV2 = lambda *args, **kwargs: None
    sys.modules["albumentations.pytorch"] = alb_pt_stub

spec = importlib.util.spec_from_file_location("utils_low_bin_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def select(medians, fractions=None, creatinine=None):
    return utils.select_low_albumin_bin_3_10_30(
        median_de_low_by_class=medians,
        low_pixel_fraction_by_class=fractions,
        creatinine_mg_dl=creatinine,
        branch_name="test_branch",
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
        {3: 0.02, 10: 0.03, 30: 0.22},
    )
    assert out["selected_low_bin"] == 30.0
    assert out["clear_30_evidence"] is True


def test_ambiguous_30_downgraded_to_10_due_to_uacr_boundary_risk():
    out = select(
        {3: 5.4, 10: 5.2, 30: 5.0},
        {3: 0.09, 10: 0.10, 30: 0.11},
        creatinine=80,
    )
    assert out["selected_low_bin"] == 10.0
    assert out["low_30_uacr_boundary_risk"] is True
    assert out["selection_reason"] == "ambiguous_30_downgraded_due_to_uacr_boundary_risk"


def test_ambiguous_30_allowed_when_clear_30_support():
    out = select(
        {3: 6.0, 10: 5.8, 30: 5.2},
        {3: 0.05, 10: 0.06, 30: 0.22},
        creatinine=80,
    )
    assert out["selected_low_bin"] == 30.0
    assert out["clear_30_evidence"] is True


def test_ambiguous_low_defaults_to_10():
    out = select(
        {3: 5.1, 10: 5.0, 30: 5.2},
        {3: 0.10, 10: 0.10, 30: 0.10},
    )
    assert out["selected_low_bin"] == 10.0
    assert out["selection_reason"] == "ambiguous_default_low_bin"


def test_ambiguous_but_3_clearly_better_than_10():
    out = select(
        {3: 4.7, 10: 5.4, 30: 5.1},
        {3: 0.10, 10: 0.10, 30: 0.10},
    )
    assert out["selected_low_bin"] == 3.0


def test_pixel_support_tiebreaker():
    out = select(
        {3: 5.0, 10: 5.1, 30: 5.2},
        {3: 0.20, 10: 0.11, 30: 0.10},
    )
    assert out["selected_low_bin"] == 3.0
    assert out["selection_reason"] == "pixel_support_tiebreak"


def test_selector_result_is_json_serializable():
    out = select(
        {3: 5.4, 10: 5.2, 30: 5.0},
        {3: 0.09, 10: 0.10, 30: 0.11},
        creatinine=80,
    )
    json.dumps(out)


def test_weak_aqua_low_compatible_branch_uses_selector_for_10():
    out = utils.microalbumin_guard_from_evidence(
        current_value_float=150.0,
        low_candidate_class_mg_l=30,
        low_shade_confirmed_relaxed=True,
        weak_aqua_low_compatible=True,
        very_low_moderate_aqua=False,
        moderate_aqua_present=False,
        strong_aqua_confirmed=False,
        median_de_low_by_class={3: 5.1, 10: 5.0, 30: 5.2},
        low_pixel_fraction_by_class={3: 0.10, 10: 0.10, 30: 0.10},
        creatinine_mg_dl=80,
    )
    assert out["action"] == "weak_aqua_low_compatible_mapped_to_low"
    assert out["corrected_albumin_value"] == 10.0


def test_very_low_moderate_branch_uses_selector_for_3():
    out = utils.microalbumin_guard_from_evidence(
        current_value_float=150.0,
        low_candidate_class_mg_l=30,
        low_shade_confirmed_relaxed=True,
        weak_aqua_low_compatible=False,
        very_low_moderate_aqua=True,
        moderate_aqua_present=False,
        strong_aqua_confirmed=False,
        median_de_low_by_class={3: 4.7, 10: 5.4, 30: 5.1},
        low_pixel_fraction_by_class={3: 0.10, 10: 0.10, 30: 0.10},
    )
    assert out["action"] == "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
    assert out["corrected_albumin_value"] == 3.0


def test_moderate_aqua_branch_unchanged():
    out = utils.microalbumin_guard_from_evidence(
        current_value_float=150.0,
        low_shade_confirmed_relaxed=False,
        moderate_aqua_present=True,
        strong_aqua_confirmed=False,
    )
    assert out["action"] == "moderate_aqua_provisional_80_150"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)
    assert out["low_bin_selection"] is None


def test_strong_aqua_branch_unchanged():
    out = utils.microalbumin_guard_from_evidence(
        current_value_float=150.0,
        strong_aqua_confirmed=True,
        moderate_aqua_present=False,
    )
    assert out["action"] == "strong_aqua_preserved_for_full_guard_matching"
    assert out["low_bin_selection"] is None
