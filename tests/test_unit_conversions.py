import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "app" / "utils.py").read_text()
start = source.index("def _safe_positive_float")
end = source.index("\ndef derive_microalbumin_guarded_uacr_scenario", start)
namespace = {"math": math}
exec(source[start:end], namespace)

convert_creatinine_to_si = namespace["convert_creatinine_to_si"]
convert_microalbumin_to_si = namespace["convert_microalbumin_to_si"]
calculate_acr_si = namespace["calculate_acr_si"]
calculate_acr_si_range = namespace["calculate_acr_si_range"]


def test_creatinine_1_mg_dl_si_units():
    out = convert_creatinine_to_si(1)
    assert out["creatinine_umol_l"] == pytest.approx(88.4)
    assert out["creatinine_mmol_l"] == pytest.approx(0.0884)


def test_microalbumin_80_mg_l_si_units():
    out = convert_microalbumin_to_si(80)
    assert out["microalbumin_g_l"] == pytest.approx(0.080)
    assert out["microalbumin_ug_ml"] == pytest.approx(80)


def test_exact_uacr_and_acr_si_stage_a2():
    uacr = round(100.0 * 80 / 150, 2)
    acr_si = calculate_acr_si(80, 150)
    assert uacr == pytest.approx(53.33)
    assert acr_si["acr_mg_mmol"] == pytest.approx(6.03)
    assert acr_si["acr_si_stage_code"] == "A2"
    assert "Moderately increased" in acr_si["acr_si_stage"]


def test_acr_si_range_stage_provisional_a2():
    out = calculate_acr_si_range((80, 150), 150)
    assert out["acr_si_range_mg_mmol"] == pytest.approx((6.03, 11.31))
    assert out["acr_si_range_display"] == "6.03–11.31 mg/mmol"
    assert out["acr_si_stage_code"] == "A2_provisional"


def test_unconfirmed_acr_si_display():
    out = calculate_acr_si(None, 150)
    assert out["acr_si_display"] == "Unconfirmed / retest"
    assert out["acr_si_stage_code"] == "unconfirmed"
