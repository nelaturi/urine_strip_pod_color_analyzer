import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _dep in ("numpy", "PIL", "cv2", "torch", "albumentations", "skimage", "joblib"):
    if importlib.util.find_spec(_dep) is None:
        pytest.skip(f"{_dep} is not installed", allow_module_level=True)

from app.utils import (
    calculate_acr_si,
    calculate_uacr_and_category,
    resolve_microalbumin_range_to_exact_if_supported,
)


def _guard(range_tuple, nearest, de, margin, mode="provisional_range", action="moderate_aqua_provisional_80_150"):
    return {
        "report_mode": mode,
        "action": action,
        "provisional_albumin_range_mg_l": range_tuple,
        "nearest_chart_bin": nearest,
        "nearest_chart_de": de,
        "nearest_vs_second_margin": margin,
    }


def test_80_150_strong_nearest_80_resolves_to_80():
    out = resolve_microalbumin_range_to_exact_if_supported(
        _guard((80.0, 150.0), 80, 7.0, 2.0),
        pixel_fraction_by_bin={80: 0.30},
        current_albumin_value=120.0,
    )
    assert out["report_mode"] == "guarded_exact"
    assert out["corrected_albumin_value"] == 80.0
    assert out["provisional_albumin_range_mg_l"] is None
    assert out["action"] == "provisional_range_resolved_to_exact"
    assert out["range_resolution"]["resolved"] is True


def test_80_150_ambiguous_evidence_remains_range():
    out = resolve_microalbumin_range_to_exact_if_supported(
        _guard((80.0, 150.0), 80, 7.0, 0.5),
        pixel_fraction_by_bin={80: 0.30},
    )
    assert out["report_mode"] == "provisional_range"
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)
    assert out["range_resolution"]["resolved"] is False


def test_150_400_strong_nearest_250_resolves_to_250():
    out = resolve_microalbumin_range_to_exact_if_supported(
        _guard((150.0, 400.0), 250, 8.5, 1.5, action="manual_provisional_150_400"),
        pixel_fraction_by_bin={250: 0.22},
    )
    assert out["corrected_albumin_value"] == 250.0
    assert out["range_resolution"]["resolved"] is True


def test_unconfirmed_ood_does_not_resolve():
    out = resolve_microalbumin_range_to_exact_if_supported(
        _guard((80.0, 150.0), 80, 6.0, 2.0, mode="unconfirmed", action="unconfirmed_ood_below_400"),
        pixel_fraction_by_bin={80: 0.4},
    )
    assert out["report_mode"] == "unconfirmed"
    assert out["range_resolution"]["resolved"] is False


def test_conventional_uacr_and_si_acr_formulas_unchanged():
    uacr, *_ = calculate_uacr_and_category(80.0, 100.0)
    acr_si = calculate_acr_si(80.0, 100.0)
    assert uacr == 80.0
    assert acr_si["acr_mg_mmol"] == round(80.0 / (100.0 * 0.0884), 2)
