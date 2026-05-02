import sys
import types
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
    cv2_stub.COLOR_RGB2HSV = 0
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 0
    cv2_stub.INTER_NEAREST = 0
    cv2_stub.erode = lambda mask_u8, kernel, iterations=1: (mask_u8 > 0).astype(np.uint8)
    sys.modules["cv2"] = cv2_stub

from app.utils import (
    choose_microalbumin_report_display,
    derive_microalbumin_guarded_uacr_scenario,
)


def test_overbright_weak_aqua_cr100_a2():
    guard = {
        "action": "unconfirmed_very_high_overbright_not_chart_like",
        "weak_aqua_present": True,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": True,
        "low_shade_confirmed": False,
        "low_candidate_class_mg_l": 10,
        "low_candidate_visual_name": "very pale grey-green",
        "median_L": 88.9,
        "low_pixel_fraction": 0.0,
        "aqua_pixel_fraction": 0.053,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 100)
    assert out["provisional_available"] is True
    assert out["provisional_albumin_range_mg_l"] == (30.0, 80.0)
    assert out["provisional_uacr_range_mg_g"] == (30.0, 80.0)
    assert out["provisional_uacr_stage_code"] == "A2_provisional"
    assert out["non_a3_supported"] is True


def test_overbright_weak_aqua_cr150_a1a2_boundary():
    guard = {
        "action": "unconfirmed_very_high_overbright_not_chart_like",
        "weak_aqua_present": True,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": True,
        "low_shade_confirmed": False,
        "low_candidate_class_mg_l": 10,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 150)
    assert out["provisional_albumin_range_mg_l"] == (30.0, 80.0)
    assert out["provisional_uacr_range_mg_g"] == (20.0, 53.33)
    assert out["provisional_uacr_stage_code"] == "A1_A2_boundary_provisional"


def test_chart_like_weak_aqua_ambiguous_250_cr150_a2():
    guard = {
        "action": "unchanged_high_ambiguous_not_confirmed",
        "weak_aqua_present": True,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": False,
        "low_shade_confirmed": False,
        "low_candidate_class_mg_l": 30,
        "low_candidate_visual_name": "pale yellow-grey-green",
        "median_L": 73.9,
        "low_pixel_fraction": 0.0,
        "aqua_pixel_fraction": 0.051,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 150)
    assert out["provisional_albumin_range_mg_l"] == (80.0, 150.0)
    assert out["provisional_uacr_range_mg_g"] == (53.33, 100.0)
    assert out["provisional_uacr_stage_code"] == "A2_provisional"
    assert out["report_mode"] == "provisional_range"


def test_overbright_no_aqua_gives_10_30():
    guard = {
        "action": "unconfirmed_very_high_overbright_not_chart_like",
        "weak_aqua_present": False,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": True,
        "low_shade_confirmed": False,
        "low_candidate_class_mg_l": 10,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 100)
    assert out["provisional_albumin_range_mg_l"] == (10.0, 30.0)
    assert out["provisional_uacr_range_mg_g"] == (10.0, 30.0)
    assert out["provisional_uacr_stage_code"] == "A1_A2_boundary_provisional"


def test_confirmed_low_exact():
    guard = {
        "action": "confirmed_low_nearest_shade",
        "weak_aqua_present": False,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": False,
        "low_shade_confirmed": True,
        "low_candidate_class_mg_l": 10,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 100)
    assert out["provisional_albumin_range_mg_l"] == (10.0, 10.0)
    assert out["provisional_uacr_range_mg_g"] == (10.0, 10.0)
    assert out["provisional_uacr_stage_code"] == "A1_provisional"
    assert out["report_mode"] == "exact_low_confirmed"


def test_no_safe_evidence():
    guard = {
        "action": "unconfirmed_very_high_not_validated_by_chart_refs",
        "weak_aqua_present": False,
        "strong_aqua_confirmed": False,
        "overbright_not_chart_like": False,
        "low_shade_confirmed": False,
    }
    out = derive_microalbumin_guarded_uacr_scenario(guard, 100)
    assert out["provisional_available"] is False
    assert out["provisional_guard_scenario"] == "no_safe_a1_a2_mapping"


def test_selector_ambiguous_does_not_show_exact_250():
    shade_guard = {"action": "unchanged_high_ambiguous_not_confirmed"}
    scenario = {"provisional_available": True, "provisional_albumin_range_mg_l": (80, 150), "provisional_albumin_display": "80–150 mg/L"}
    out = choose_microalbumin_report_display(250, shade_guard, scenario)
    assert out["microalbumin_report_mode"] == "provisional_range"
    assert "80" in out["microalbumin_display_text"]
    assert "150" in out["microalbumin_display_text"]
    assert out["microalbumin_display_text"] != "250 mg/L"


def test_selector_unconfirmed_does_not_restore_1000():
    shade_guard = {"action": "unconfirmed_very_high_overbright_not_chart_like"}
    scenario = {"provisional_available": True, "provisional_albumin_range_mg_l": (30, 80), "provisional_albumin_display": "30–80 mg/L"}
    out = choose_microalbumin_report_display(None, shade_guard, scenario)
    assert out["microalbumin_report_mode"] == "provisional_range"
    assert "30" in out["microalbumin_display_text"]
    assert "80" in out["microalbumin_display_text"]
    assert "1000" not in out["microalbumin_display_text"]
