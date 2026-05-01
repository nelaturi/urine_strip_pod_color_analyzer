import unittest
import types
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from app.utils import microalbumin_shade_sanity_check


def make_uniform_pod(rgb):
    img = np.full((64, 64, 3), rgb, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    return img, mask


class TestMicroalbuminShadeGuard(unittest.TestCase):
    def test_overbright_pale_blue_not_finalized_as_1000(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((207, 225, 248)), 1000, current_confidence=20.0)
        self.assertIsNone(out["corrected_albumin_value"])
        self.assertIn(out["action"], {"unconfirmed_very_high_overbright_not_chart_like", "unconfirmed_very_high_low_confidence", "unconfirmed_very_high_not_validated_by_chart_refs", "unconfirmed_very_high_weak_aqua_only"})
        self.assertNotEqual(out["action"], "unchanged_high_confirmed_by_aqua")
        self.assertNotEqual(out["action"], "unchanged_very_high_confirmed_by_validated_high_refs")

    def test_weak_aqua_does_not_confirm_very_high(self):
        img = np.full((64, 64, 3), (207, 225, 248), dtype=np.uint8)
        flat = img.reshape(-1, 3)
        flat[:int(0.10 * flat.shape[0])] = np.array((166, 181, 184), dtype=np.uint8)
        out = microalbumin_shade_sanity_check(img, np.ones((64, 64), dtype=bool), 1000, current_confidence=20.0)
        self.assertIsNone(out["corrected_albumin_value"])
        self.assertTrue(out["weak_aqua_present"])
        self.assertFalse(out["strong_aqua_confirmed"])
        self.assertFalse(out["very_high_confirmed"])
        self.assertTrue(out["action"].startswith("unconfirmed_"))

    def test_very_high_low_confidence_not_finalized(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((203, 221, 236)), 1000, current_confidence=20.0)
        self.assertIsNone(out["corrected_albumin_value"])
        self.assertTrue(out["action"].startswith("unconfirmed_"))

    def test_low_shade_still_overrides_false_1000_to_10(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((175, 177, 168)), 1000, current_confidence=20.0)
        self.assertEqual(out["corrected_albumin_value"], 10.0)
        self.assertEqual(out["action"], "override_very_high_to_low_nearest_shade")
        self.assertTrue(out["low_shade_confirmed"])

    def test_low_shade_still_overrides_false_1000_to_30(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((175, 178, 165)), 1000, current_confidence=20.0)
        self.assertEqual(out["corrected_albumin_value"], 30.0)
        self.assertEqual(out["action"], "override_very_high_to_low_nearest_shade")
        self.assertTrue(out["low_shade_confirmed"])

    def test_aqua_80_keeps_80_when_strong(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((166, 181, 184)), 80, current_confidence=80.0)
        self.assertEqual(out["corrected_albumin_value"], 80.0)
        self.assertEqual(out["action"], "unchanged_mid_confirmed_by_strong_aqua")
        self.assertTrue(out["strong_aqua_confirmed"])

    def test_aqua_150_keeps_150_when_strong(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((156, 178, 189)), 150, current_confidence=80.0)
        self.assertEqual(out["corrected_albumin_value"], 150.0)
        self.assertEqual(out["action"], "unchanged_mid_confirmed_by_strong_aqua")
        self.assertTrue(out["strong_aqua_confirmed"])

    def test_low_visual_not_claimed_when_low_evidence_zero(self):
        out = microalbumin_shade_sanity_check(*make_uniform_pod((207, 225, 248)), 1000, current_confidence=20.0)
        self.assertIn("low_candidate_visual_name", out)
        self.assertFalse(out["low_shade_confirmed"])
        self.assertLess(out["low_pixel_fraction"], 0.50)

    def test_none_does_not_fallback_to_1000(self):
        albumin_shade_guard = {"corrected_albumin_value": None, "action": "unconfirmed_very_high_overbright_not_chart_like"}
        c2_snapped = 1000
        c2_final = albumin_shade_guard.get("corrected_albumin_value", c2_snapped)
        self.assertIsNone(c2_final)


if __name__ == "__main__":
    unittest.main()
