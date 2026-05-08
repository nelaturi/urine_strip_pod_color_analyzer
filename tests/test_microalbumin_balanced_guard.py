import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Keep this shade-guard unit test independent of the web/model stack.
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
    def _erode(mask_u8, kernel, iterations=1):
        out = (mask_u8 > 0).astype(np.uint8)
        for _ in range(iterations):
            padded = np.pad(out, ((1, 1), (1, 1)), mode="constant")
            nxt = np.zeros_like(out)
            for y in range(out.shape[0]):
                for x in range(out.shape[1]):
                    nxt[y, x] = 1 if np.all(padded[y:y + 3, x:x + 3] > 0) else 0
            out = nxt
        return out
    cv2_stub.erode = _erode
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
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        out = np.empty(arr.shape, dtype=np.float64)
        out[..., 0] = (r + g + b) / (3.0 * 255.0) * 100.0
        out[..., 1] = (g - r) * 0.35
        out[..., 2] = (b - g) * 0.35
        return out
    def _deltae(a, b):
        return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)
    color_stub.rgb2lab = _rgb2lab
    color_stub.deltaE_ciede2000 = _deltae
    skimage_stub.color = color_stub
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.color"] = color_stub

spec = importlib.util.spec_from_file_location("utils_under_test", ROOT / "app" / "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

MICROALBUMIN_CENTROIDS = utils.MICROALBUMIN_CENTROIDS
microalbumin_shade_sanity_check = utils.microalbumin_shade_sanity_check
calculate_uacr_range_and_stage = utils.calculate_uacr_range_and_stage


def make_uniform_pod(rgb):
    img = np.full((64, 64, 3), rgb, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    return img, mask


def test_true_low_3_maps_to_3():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[3]), 1000)
    assert out["corrected_albumin_value"] == 3.0
    assert out["low_shade_confirmed"] is True


def test_true_low_10_maps_to_10():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[10]), 1000)
    assert out["corrected_albumin_value"] == 10.0


def test_true_low_30_maps_to_30():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[30]), 1000)
    assert out["corrected_albumin_value"] == 30.0


def test_overbright_no_aqua_maps_to_guarded_10():
    out = microalbumin_shade_sanity_check(*make_uniform_pod((245, 245, 245)), 1000)
    assert out["action"] == "guarded_overbright_no_aqua_to_10"
    assert out["corrected_albumin_value"] == 10.0
    assert out["weak_aqua_present"] is False


def test_overbright_weak_aqua_maps_to_guarded_80():
    img = np.full((64, 64, 3), (245, 245, 245), dtype=np.uint8)
    flat = img.reshape(-1, 3)
    flat[: int(0.06 * flat.shape[0])] = np.array(MICROALBUMIN_CENTROIDS[150], dtype=np.uint8)
    out = microalbumin_shade_sanity_check(img, np.ones((64, 64), dtype=bool), 1000)
    assert out["action"] == "guarded_overbright_weak_aqua_to_80"
    assert out["corrected_albumin_value"] == 80.0


def test_chart_like_weak_aqua_gives_provisional_80_150():
    img = np.full((64, 64, 3), (130, 130, 130), dtype=np.uint8)
    flat = img.reshape(-1, 3)
    flat[: int(0.06 * flat.shape[0])] = np.array(MICROALBUMIN_CENTROIDS[150], dtype=np.uint8)
    out = microalbumin_shade_sanity_check(img, np.ones((64, 64), dtype=bool), 1000)
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)


def test_strong_aqua_at_150_maps_to_150():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[150]), 1000)
    assert out["strong_aqua_confirmed"] is True
    assert out["corrected_albumin_value"] == 150.0


def test_strong_aqua_near_250_with_confidence_maps_to_250():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[250]), 1000, current_confidence=0.70)
    assert out["corrected_albumin_value"] == 250.0


def test_strong_aqua_near_400_with_confidence_maps_to_400():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[400]), 1000, current_confidence=0.70)
    assert out["corrected_albumin_value"] == 400.0


def test_strong_aqua_near_600_low_confidence_becomes_provisional_150_400():
    out = microalbumin_shade_sanity_check(*make_uniform_pod(MICROALBUMIN_CENTROIDS[600]), 1000, current_confidence=0.50)
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (150.0, 400.0)


def test_current_1000_without_low_or_strong_aqua_becomes_unconfirmed():
    out = microalbumin_shade_sanity_check(*make_uniform_pod((120, 120, 120)), 1000)
    assert out["report_mode"] == "unconfirmed"
    assert out["corrected_albumin_value"] is None
    assert out["action"] == "unconfirmed_very_high_without_validated_support"


def test_current_600_low_confidence_becomes_provisional_150_400():
    out = microalbumin_shade_sanity_check(*make_uniform_pod((120, 120, 120)), 600, current_confidence=0.50)
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (150.0, 400.0)


def test_uacr_range_helper_works():
    out = calculate_uacr_range_and_stage((80, 150), 150)
    assert out["uacr_range_mg_g"] == (53.33, 100.0)
    assert out["uacr_stage_code"] == "A2_provisional"
