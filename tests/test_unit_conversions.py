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

spec = importlib.util.spec_from_file_location("utils_unit_conversion_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def test_creatinine_1_mg_dl_si_conversion():
    converted = utils.convert_creatinine_to_si(1)
    assert converted["creatinine_umol_l"] == pytest.approx(88.4)
    assert converted["creatinine_mmol_l"] == pytest.approx(0.0884)


def test_microalbumin_80_mg_l_si_and_equivalent_units():
    converted = utils.convert_microalbumin_to_si(80)
    assert converted["microalbumin_g_l"] == pytest.approx(0.080)
    assert converted["microalbumin_ug_ml"] == pytest.approx(80)
    assert converted["microalbumin_si_display"] == "0.080 g/L"


def test_albumin_80_creatinine_150_uacr_and_acr_si_stage_a2():
    uacr_value, conventional_stage, _, _ = utils.calculate_uacr_and_category(80, 150)
    acr_si = utils.calculate_acr_si(80, 150)
    assert uacr_value == pytest.approx(53.33)
    assert conventional_stage == "A2 Proteinuria"
    assert acr_si["acr_mg_mmol"] == pytest.approx(6.03)
    assert acr_si["acr_si_stage_code"] == "A2"


def test_albumin_80_150_range_creatinine_150_acr_si_range_provisional_a2():
    acr_range = utils.calculate_acr_si_range((80, 150), 150)
    assert acr_range["acr_si_range_mg_mmol"] == pytest.approx((6.03, 11.31))
    assert acr_range["acr_si_range_display"] == "6.03–11.31 mg/mmol"
    assert acr_range["acr_si_stage_code"] == "A2_provisional"
    assert acr_range["acr_si_stage"] == "Provisional A2 / retest"


def test_unconfirmed_albumin_acr_si_display():
    acr_si = utils.calculate_acr_si(None, 150)
    assert acr_si["acr_si_display"] == "Unconfirmed / retest"
    assert acr_si["acr_si_stage_code"] == "unconfirmed"
