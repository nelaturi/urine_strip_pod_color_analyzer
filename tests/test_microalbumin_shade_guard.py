import unittest
import types
import sys

import numpy as np

# Lightweight module stubs to keep tests independent of OpenCV/albumentations runtime libs.
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
    def test_a_uniform_10_shade_overrides_800(self):
        img, mask = make_uniform_pod((175, 177, 168))
        out = microalbumin_shade_sanity_check(img, mask, 800)
        self.assertTrue(out["guard_applied"])
        self.assertEqual(out["corrected_albumin_value"], 10.0)
        self.assertEqual(out["action"], "override_high_to_low_nearest_shade")
        self.assertLess(out["aqua_pixel_fraction"], 0.05)

    def test_b_uniform_30_shade_overrides_1000(self):
        img, mask = make_uniform_pod((175, 178, 165))
        out = microalbumin_shade_sanity_check(img, mask, 1000)
        self.assertTrue(out["guard_applied"])
        self.assertEqual(out["corrected_albumin_value"], 30.0)
        self.assertEqual(out["action"], "override_high_to_low_nearest_shade")
        self.assertLess(out["aqua_pixel_fraction"], 0.05)

    def test_c_uniform_10_shade_with_low_value_confirms_low(self):
        img, mask = make_uniform_pod((175, 177, 168))
        out = microalbumin_shade_sanity_check(img, mask, 10)
        self.assertFalse(out["guard_applied"])
        self.assertIn(out["corrected_albumin_value"], [10.0, 30.0])
        self.assertEqual(out["action"], "confirmed_low_nearest_shade")

    def test_d_uniform_30_shade_with_low_value_confirms_low(self):
        img, mask = make_uniform_pod((175, 178, 165))
        out = microalbumin_shade_sanity_check(img, mask, 30)
        self.assertFalse(out["guard_applied"])
        self.assertIn(out["corrected_albumin_value"], [10.0, 30.0])
        self.assertEqual(out["action"], "confirmed_low_nearest_shade")

    def test_e_uniform_80_aqua_confirms_high(self):
        img, mask = make_uniform_pod((166, 181, 184))
        out = microalbumin_shade_sanity_check(img, mask, 80)
        self.assertFalse(out["guard_applied"])
        self.assertEqual(out["corrected_albumin_value"], 80.0)
        self.assertTrue(out["high_value_confirmed_by_aqua"])
        self.assertEqual(out["action"], "unchanged_high_confirmed_by_aqua")

    def test_f_uniform_150_aqua_confirms_high(self):
        img, mask = make_uniform_pod((156, 178, 189))
        out = microalbumin_shade_sanity_check(img, mask, 150)
        self.assertFalse(out["guard_applied"])
        self.assertEqual(out["corrected_albumin_value"], 150.0)
        self.assertTrue(out["high_value_confirmed_by_aqua"])
        self.assertEqual(out["action"], "unchanged_high_confirmed_by_aqua")

    def test_g_uniform_10_shade_does_not_finalize_1000(self):
        img, mask = make_uniform_pod((175, 177, 168))
        out = microalbumin_shade_sanity_check(img, mask, 1000)
        self.assertNotEqual(out["corrected_albumin_value"], 1000)
        self.assertIn(out["corrected_albumin_value"], [10.0, 30.0])

    def test_h_noisy_low_shade_still_overrides_high(self):
        rng = np.random.default_rng(7)
        img = np.full((64, 64, 3), (175, 178, 165), dtype=np.int16)
        noisy_count = int(0.10 * 64 * 64)
        flat_idx = rng.choice(64 * 64, size=noisy_count, replace=False)
        noise = rng.integers(-3, 4, size=(noisy_count, 3))
        flat = img.reshape(-1, 3)
        flat[flat_idx] = np.clip(flat[flat_idx] + noise, 0, 255)
        img_u8 = flat.reshape(64, 64, 3).astype(np.uint8)
        mask = np.ones((64, 64), dtype=bool)

        out = microalbumin_shade_sanity_check(img_u8, mask, 800)
        self.assertTrue(out["guard_applied"])
        self.assertIn(out["corrected_albumin_value"], [10.0, 30.0])

    def test_i_mixed_with_aqua_fraction_blocks_override(self):
        img = np.full((64, 64, 3), (175, 178, 165), dtype=np.uint8)
        aqua_count = int(0.15 * 64 * 64)
        flat = img.reshape(-1, 3)
        flat[:aqua_count] = np.array((166, 181, 184), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)

        out = microalbumin_shade_sanity_check(img, mask, 80)
        self.assertFalse(out["guard_applied"])
        self.assertEqual(out["action"], "unchanged_high_confirmed_by_aqua")

    def test_j_periwinkle_blue_cast_1000_is_clipped_to_mid_range(self):
        img, mask = make_uniform_pod((152, 162, 209))
        out = microalbumin_shade_sanity_check(img, mask, 1000)
        self.assertTrue(out["guard_applied"])
        self.assertEqual(out["action"], "override_high_to_mid_cool_blue_cast")
        self.assertTrue(out["likely_cool_blue_cast"])
        self.assertLessEqual(out["corrected_albumin_value"], 600.0)
        self.assertGreaterEqual(out["median_blue_minus_green"], 24.0)

    def test_k_lavender_blue_cast_1000_is_clipped_to_mid_range(self):
        img, mask = make_uniform_pod((152, 168, 247))
        out = microalbumin_shade_sanity_check(img, mask, 1000)
        self.assertTrue(out["guard_applied"])
        self.assertEqual(out["action"], "override_high_to_mid_cool_blue_cast")
        self.assertTrue(out["likely_cool_blue_cast"])
        self.assertLessEqual(out["corrected_albumin_value"], 600.0)
        self.assertLess(out["aqua_pixel_fraction"], 0.05)

    def test_l_blue_cast_800_is_clipped_when_not_aqua(self):
        img, mask = make_uniform_pod((138, 163, 228))
        out = microalbumin_shade_sanity_check(img, mask, 800)
        self.assertTrue(out["guard_applied"])
        self.assertEqual(out["action"], "override_high_to_mid_cool_blue_cast")
        self.assertTrue(out["likely_cool_blue_cast"])
        self.assertLessEqual(out["corrected_albumin_value"], 600.0)


if __name__ == "__main__":
    unittest.main()
