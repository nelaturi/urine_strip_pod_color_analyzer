import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from app.utils import microalbumin_shade_sanity_check


def make_uniform_pod(rgb):
    img = np.full((64, 64, 3), rgb, dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    return img, mask


def test_low_10_not_finalized_as_800():
    img, mask = make_uniform_pod((175, 177, 168))
    out = microalbumin_shade_sanity_check(img, mask, 800)
    assert out["guard_applied"] is True
    assert out["corrected_albumin_value"] == 10.0
    assert out["action"] == "override_high_to_low_nearest_shade"
    assert out["aqua_pixel_fraction"] < 0.05


def test_low_30_not_finalized_as_1000():
    img, mask = make_uniform_pod((175, 178, 165))
    out = microalbumin_shade_sanity_check(img, mask, 1000)
    assert out["guard_applied"] is True
    assert out["corrected_albumin_value"] == 30.0
    assert out["action"] == "override_high_to_low_nearest_shade"
    assert out["aqua_pixel_fraction"] < 0.05


def test_existing_low_10_confirmed():
    img, mask = make_uniform_pod((175, 177, 168))
    out = microalbumin_shade_sanity_check(img, mask, 10)
    assert out["guard_applied"] is False
    assert out["corrected_albumin_value"] in [10.0, 30.0]
    assert out["action"] == "confirmed_low_nearest_shade"


def test_existing_low_30_confirmed():
    img, mask = make_uniform_pod((175, 178, 165))
    out = microalbumin_shade_sanity_check(img, mask, 30)
    assert out["guard_applied"] is False
    assert out["corrected_albumin_value"] in [10.0, 30.0]
    assert out["action"] == "confirmed_low_nearest_shade"


def test_aqua_80_blocks_downward_override():
    img, mask = make_uniform_pod((166, 181, 184))
    out = microalbumin_shade_sanity_check(img, mask, 80)
    assert out["guard_applied"] is False
    assert out["corrected_albumin_value"] == 80.0
    assert out["high_value_confirmed_by_aqua"] is True
    assert out["action"] == "unchanged_high_confirmed_by_aqua"


def test_aqua_150_blocks_downward_override():
    img, mask = make_uniform_pod((156, 178, 189))
    out = microalbumin_shade_sanity_check(img, mask, 150)
    assert out["guard_applied"] is False
    assert out["corrected_albumin_value"] == 150.0
    assert out["high_value_confirmed_by_aqua"] is True
    assert out["action"] == "unchanged_high_confirmed_by_aqua"


def test_low_30_with_noise_not_finalized_as_800():
    rng = np.random.default_rng(7)
    base = np.full((64, 64, 3), (175, 178, 165), dtype=np.int16)
    noise = rng.integers(-3, 4, size=(64, 64, 3), dtype=np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    out = microalbumin_shade_sanity_check(img, mask, 800)
    assert out["guard_applied"] is True
    assert out["corrected_albumin_value"] in [10.0, 30.0]


def test_aqua_fraction_blocks_low_override():
    img = np.full((64, 64, 3), (175, 178, 165), dtype=np.uint8)
    img[:8, :, :] = np.array((166, 181, 184), dtype=np.uint8)  # 12.5%
    mask = np.ones((64, 64), dtype=bool)
    out = microalbumin_shade_sanity_check(img, mask, 80)
    assert out["guard_applied"] is False
    assert out["corrected_albumin_value"] == 80.0
    assert out["action"] == "unchanged_high_confirmed_by_aqua"
