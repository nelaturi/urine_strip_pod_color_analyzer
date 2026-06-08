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
    MICRO_VERY_HIGH_DISPLAY_MODE,
    calculate_uacr_and_category,
    choose_microalbumin_report_display,
)


def test_albumin_1500_displays_exact_value_only():
    display = choose_microalbumin_report_display(
        1500.0,
        {"report_mode": "high_watch", "action": "high_value_verified_preserved"},
        {},
    )
    assert display["microalbumin_report_mode"] == MICRO_VERY_HIGH_DISPLAY_MODE
    assert display["microalbumin_exact_value_mg_l"] == 1500.0
    assert display["microalbumin_display_text"] == "1500 mg/L"
    assert "Unconfirmed" not in display["microalbumin_display_text"]
    assert "Retest" not in display["microalbumin_display_text"]
    assert display["microalbumin_very_high_display"]["triggered"] is True


def test_albumin_1300_does_not_trigger_override():
    display = choose_microalbumin_report_display(
        1300.0,
        {"report_mode": "high_watch", "action": "high_value_verified_preserved"},
        {},
    )
    assert display["microalbumin_report_mode"] == "high_watch"
    assert display["microalbumin_very_high_display"]["triggered"] is False


def test_uacr_computed_only_with_valid_creatinine():
    valid_uacr, *_ = calculate_uacr_and_category(1500.0, 100.0)
    invalid_uacr, *_ = calculate_uacr_and_category(1500.0, None)
    zero_uacr, *_ = calculate_uacr_and_category(1500.0, 0.0)
    assert valid_uacr == 1500.0
    assert invalid_uacr is None
    assert zero_uacr is None
