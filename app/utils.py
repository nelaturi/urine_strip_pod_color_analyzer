import logging
import matplotlib
matplotlib.use('Agg') # MUST be before importing pyplot

import os
import cv2
import uuid
import sys
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt # After matplotlib.use
import joblib
from skimage.color import rgb2lab
import math

# Try to use ΔE2000; fall back to CIE76 if not available
try:
    from skimage.color import deltaE_ciede2000 as _deltaE2000
    HAVE_DE00 = True
except Exception:
    HAVE_DE00 = False

# ───────────────────────────────
# Constants & Configuration
# ───────────────────────────────

# Segmentation indices: Pod1=Creatinine, Pod2=Microalbumin
POD1_IDX, POD2_IDX = 4, 5

# Distance/override thresholds (tune on validation)
MICRO_DELTAE_ACCEPT = 8.0     # accept color centroid if best ΔE <= this
MICRO_DELTAE_MARGIN = 3.0     # require ΔE_reg - ΔE_best > margin to override regression
POD_ERODE_KERNEL = 3          # 3x3 erosion to avoid edges
WB_EPS = 1e-6                 # numerics for white balance
LOW_END_THRESHOLDS = {
    'microalbumin': 30.0,     # mg/L
    'creatinine': 25.0,       # mg/dL
}
CREATININE_DELTAE_ACCEPT = 8.0  # accept creatinine centroid if best ΔE <= this
MICRO_LOW_DE_MAX = 7.5
MICRO_LOW_PIXEL_FRACTION_MIN = 0.40
MICRO_LOW_MARGIN_AMBIGUOUS = 1.2

# Below-400 balanced guard
MICRO_LOW_DE_RELAXED_MAX = 11.5
MICRO_LOW_PIXEL_FRACTION_RELAXED_MIN = 0.07
MICRO_LOW_VS_AQUA_MARGIN_MIN = 0.4

# New faint weak aqua tier: low-compatible
MICRO_WEAK_AQUA_LOW_MIN = 0.03
MICRO_WEAK_AQUA_LOW_MAX = 0.22
MICRO_WEAK_AQUA_LOW_DE_MAX = 15.0
MICRO_WEAK_AQUA_LOW_LOW_SUPPORT_MIN = 0.07
MICRO_WEAK_AQUA_LOW_MARGIN_TOL = 2.0

# Rename/reinterpret old weak aqua as moderate aqua
MICRO_MODERATE_AQUA_MIN = 0.25
MICRO_MODERATE_AQUA_DE_MAX = 13.0
MICRO_MODERATE_AQUA_LOW_SUPPRESSION_MAX = 0.06

MICRO_STRONG_AQUA_MIN = 0.30
MICRO_STRONG_AQUA_DE_MAX = 11.0
MICRO_AQUA_ADVANTAGE_MARGIN = 1.5

# Backward-compatible aliases for older callers/tests.
MICRO_WEAK_AQUA_MIN = MICRO_MODERATE_AQUA_MIN
MICRO_WEAK_AQUA_DE_MAX = MICRO_MODERATE_AQUA_DE_MAX

# High-value verification
MICRO_HIGH_VERIFY_MIN_VALUE = 400.0
MICRO_HIGH_VERIFY_DE_MAX = 13.0
MICRO_HIGH_VERIFY_MARGIN_MIN = 1.0
MICRO_HIGH_VERIFY_MIN_AQUA_FRACTION = 0.25

# OOD / chart-likeness
MICRO_CHARTLIKE_DE_MAX = 16.0
MICRO_OOD_PIXEL_FRACTION_MAX = 0.02

MICRO_OVERBRIGHT_L_MAX = 84.0

MICRO_HIGH_WATCH_MIN = 600.0
MICRO_VERY_HIGH_GUARD_MIN = 800.0

MICRO_MIN_CONF_FOR_250_400 = 0.65
MICRO_MIN_CONF_FOR_600 = 0.75

MICRO_SHADE_MIN_PIXELS = 50

# Legacy aliases retained for compatibility with older reporting/tests.
MICRO_SHADE_PIXEL_DE_ACCEPT = MICRO_STRONG_AQUA_DE_MAX
MICRO_SHADE_MEDIAN_LOW_DE_ACCEPT = MICRO_LOW_DE_MAX
MICRO_SHADE_ADVANTAGE_MARGIN = MICRO_AQUA_ADVANTAGE_MARGIN
MICRO_LOW_SHADE_FRACTION_MIN = MICRO_LOW_PIXEL_FRACTION_MIN
MICRO_AQUA_WEAK_FRACTION_MIN = MICRO_MODERATE_AQUA_MIN
MICRO_AQUA_STRONG_FRACTION_MIN = MICRO_STRONG_AQUA_MIN
MICRO_AQUA_VERY_HIGH_FRACTION_MIN = 0.25
MICRO_HIGH_VALUE_MIN = 150.0
MICRO_VERY_HIGH_VALUE_MIN = MICRO_VERY_HIGH_GUARD_MIN
MICRO_CHART_MEDIAN_DE_ACCEPT = MICRO_STRONG_AQUA_DE_MAX
MICRO_MIN_CONFIDENCE_FOR_VERY_HIGH = MICRO_MIN_CONF_FOR_600
MICRO_UNCONFIRMED_RETURNS_NONE = True

# Experimental legacy-recovery fallback for V4-unconfirmed microalbumin only.
ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY = True
MICRO_LEGACY_RECOVERY_MODE = "liberal"
MICRO_LEGACY_RECOVERY_MIN_CHARTLIKE_DE = 28.0
MICRO_LEGACY_RECOVERY_MIN_MARGIN = -999.0
MICRO_LEGACY_RECOVERY_ACCEPT_OOD = True
MICRO_LEGACY_RECOVERY_ACCEPT_LOW_CONFIDENCE = True
MICRO_LEGACY_RECOVERY_CONFLICT_POLICY = "average_bin"
MICRO_LEGACY_RECOVERY_UACR_POLICY = "higher_uacr"
MICRO_LEGACY_RECOVERY_ALLOWED_BINS = [
    3, 10, 30, 80, 150, 250, 400, 600, 800, 1000, 1400
]

# Post-extraction quality-control thresholds
QUALITY_THRESHOLDS = {
    'sigma_l': 14.0,
    'mad_ab': 7.0,
    'de_p90': 14.0,
    'texture_score': 16.0,
    'glare_v_high': 245,
    'glare_s_low': 45,
    'glare_l_high': 92.0,
    'glare_area_ratio': 0.04,
    'glare_cluster_ratio': 0.02,
    'glare_spot_count': 2,
    'mask_area_min': 0.002,
    'mask_area_max': 0.20,
    'edge_touch_ratio': 0.15,
    'eccentricity_max': 0.98,
}

CONFIDENCE_WEIGHTS = {
    'non_uniform': 0.45,
    'glare': 0.35,
    'mask': 0.20,
}

SOFT_HARD_THRESHOLDS = {
    'de_soft': 10.0,
    'de_hard': 22.0,
    'ga_soft': 0.01,
    'ga_hard': 0.08,
}

HYSTERESIS_BAND = {
    'pod1': 10.0,   # mg/dL
    'pod2': 20.0,   # mg/L
}

# Preprocessing pipeline (keep model input unchanged)
val_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ───────────────────────────────
# PyInstaller Compatibility
# ───────────────────────────────

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)

# ───────────────────────────────
# Load Regression Models
# ───────────────────────────────
CREATININE_MODEL_PATH = resource_path('app/model/creatinine_model.pkl')
MICROALBUMIN_MODEL_PATH = resource_path('app/model/microalbumin_model.pkl')

model_creat = None
model_micro = None
try:
    model_creat = joblib.load(CREATININE_MODEL_PATH)
    model_micro = joblib.load(MICROALBUMIN_MODEL_PATH)
    logging.info("Successfully loaded calibrated models.")
except Exception as e:
    logging.error("FATAL: Failed to load .pkl models.", exc_info=True)

# ───────────────────────────────
# Empirical RGB Centroids
# ───────────────────────────────
# Microalbumin (pod2) → new bin 3, updated bin 10
MICROALBUMIN_CENTROIDS = {
    3:    (176, 180, 165),   # New bin (from Label 3 dataset)
    10:   (177, 182, 171),   # Updated centroid from new Label 10 dataset
    30:   (160.0, 181.0, 170.0),
    80:   (162.0, 176.0, 169.0),
    150:  (153.0, 180.0, 175.0),
    250:  (132.0, 168.0, 162.0),
    400:  (140.0, 173.9, 174.1),
    600:  (125.0, 164.0, 163.0),
    800:  (138.0, 174.0, 178.0),
    1000: (134.0, 176.0, 184.0),
    1400: (158.0, 188.0, 192.0),
}

MICROALBUMIN_LOW_SHADE_REFS = {
    10: {
        "rgb": (175, 177, 168),
        "hex": "#AFB1A8",
        "visual_name": "very pale grey-green",
    },
    30: {
        "rgb": (175, 178, 165),
        "hex": "#AFB2A5",
        "visual_name": "pale yellow-grey-green",
    },
}

MICROALBUMIN_AQUA_CONFIRM_REFS = {
    80: {
        "rgb": (166, 181, 184),
        "hex": "#A6B5B8",
        "visual_name": "pale aqua / blue-grey",
    },
    150: {
        "rgb": (156, 178, 189),
        "hex": "#9CB2BD",
        "visual_name": "light aqua-blue / blue-grey",
    },
}

MICROALBUMIN_LOW_EXACT_REFS = {
    3: MICROALBUMIN_CENTROIDS[3],
    10: MICROALBUMIN_CENTROIDS[10],
    30: MICROALBUMIN_CENTROIDS[30],
}

MICROALBUMIN_LOW_EXACT_VISUAL_NAMES = {
    3: "very low pale grey-green",
    10: "low pale grey-green",
    30: "low pale yellow-grey-green",
}

MICRO_STRONG_AQUA_ALLOWED_BINS = [150, 250, 400, 600]
MICRO_VERY_HIGH_BINS = [800, 1000, 1400, 1800]


# Creatinine (pod1) → added bin 25
CREATININE_CENTROIDS = {
    "10 (0.1)":  (190, 177, 52), # Updated centroid from new Label 10 dataset
    "25 (0.25)": (184, 168, 47),  # New bin (from Label 25 dataset)
    "50 (0.5)":  (178, 167, 49),
    "100 (1.0)": (169, 168, 55),
    "150 (1.5)": (144, 152, 53),
    "200 (2.0)": (135, 145, 55),
    "300 (4.0)": (128, 64, 0)
}

# ───────────────────────────────
# Color-space helpers
# ───────────────────────────────

def _rgb_to_lab_triplet(rgb_triplet):
    arr = np.uint8([[rgb_triplet]])
    return rgb2lab(arr).reshape(3,).astype(np.float64)

def _lab_distance(lab1, lab2, wL=0.5):
    if HAVE_DE00:
        a = np.array(lab1, dtype=np.float64).reshape(1, 1, 3)
        b = np.array(lab2, dtype=np.float64).reshape(1, 1, 3)
        return float(_deltaE2000(a, b)[0, 0])
    dL = (lab1[0] - lab2[0]) * math.sqrt(max(wL, 0.0))
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return float(math.sqrt(dL*dL + da*da + db*db))

MICROALBUMIN_CENTROIDS_LAB = {k: _rgb_to_lab_triplet(v) for k, v in MICROALBUMIN_CENTROIDS.items()}
CREATININE_CENTROIDS_LAB = {k: _rgb_to_lab_triplet(v) for k, v in CREATININE_CENTROIDS.items()}
MICROALBUMIN_LOW_SHADE_REFS_LAB = {
    k: _rgb_to_lab_triplet(v["rgb"])
    for k, v in MICROALBUMIN_LOW_SHADE_REFS.items()
}
MICROALBUMIN_AQUA_CONFIRM_REFS_LAB = {
    k: _rgb_to_lab_triplet(v["rgb"])
    for k, v in MICROALBUMIN_AQUA_CONFIRM_REFS.items()
}
MICROALBUMIN_LOW_EXACT_REFS_LAB = {
    k: _rgb_to_lab_triplet(v)
    for k, v in MICROALBUMIN_LOW_EXACT_REFS.items()
}
MICRO_STRONG_AQUA_ALLOWED_REFS_LAB = {
    k: MICROALBUMIN_CENTROIDS_LAB[k]
    for k in MICRO_STRONG_AQUA_ALLOWED_BINS
    if k in MICROALBUMIN_CENTROIDS_LAB
}


def _deltae_pixels_to_lab_ref(lab_pixels, lab_ref):
    """
    lab_pixels: shape (N, 3)
    lab_ref: shape (3,)
    Return: shape (N,) ΔE distance array.
    Prefer CIEDE2000 if available. Fall back to weighted CIE76 using _lab_distance wL logic.
    """
    if lab_pixels.size == 0:
        return np.array([], dtype=np.float64)
    px = np.asarray(lab_pixels, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(lab_ref, dtype=np.float64).reshape(1, 3)
    if HAVE_DE00:
        px_reshaped = px.reshape(-1, 1, 3)
        ref_reshaped = np.tile(ref.reshape(1, 1, 3), (px.shape[0], 1, 1))
        return _deltaE2000(px_reshaped, ref_reshaped).reshape(-1).astype(np.float64)
    dL = (px[:, 0] - ref[0, 0]) * math.sqrt(0.5)
    da = px[:, 1] - ref[0, 1]
    db = px[:, 2] - ref[0, 2]
    return np.sqrt(dL * dL + da * da + db * db)


def _nearest_micro_allowed_bin_lab(lab_obs, allowed_bins):
    """
    Return nearest allowed microalbumin bin and ΔE among allowed_bins.
    Only uses bins with available LAB centroids.
    """
    best_label = None
    best_de = float("inf")
    for lbl in allowed_bins:
        if lbl not in MICROALBUMIN_CENTROIDS_LAB:
            continue
        de = _lab_distance(lab_obs, MICROALBUMIN_CENTROIDS_LAB[lbl])
        if de < best_de:
            best_label = lbl
            best_de = de
    return best_label, best_de


def _nearest_micro_chart_bin_with_margin(lab_obs):
    """
    Return nearest microalbumin chart bin, nearest ΔE, second-nearest ΔE,
    and nearest-vs-second margin using available chart LAB centroids.
    """
    distances = []
    for lbl, lab_ref in MICROALBUMIN_CENTROIDS_LAB.items():
        distances.append((int(lbl), float(_lab_distance(lab_obs, lab_ref))))
    distances.sort(key=lambda item: item[1])
    if not distances:
        return None, None, None, None
    nearest_chart_bin, nearest_chart_de = distances[0]
    second_nearest_chart_de = distances[1][1] if len(distances) > 1 else None
    nearest_vs_second_margin = (
        None
        if second_nearest_chart_de is None
        else float(second_nearest_chart_de - nearest_chart_de)
    )
    return (
        int(nearest_chart_bin),
        float(nearest_chart_de),
        None if second_nearest_chart_de is None else float(second_nearest_chart_de),
        nearest_vs_second_margin,
    )



def evaluate_microalbumin_aqua_tiers(
    aqua_pixel_fraction,
    median_aqua_de,
    low_pixel_fraction,
    median_low_de,
    strong_aqua_confirmed=False,
):
    """Classify aqua evidence tiers without changing V4 guard structure."""
    aqua_fraction = float(aqua_pixel_fraction or 0.0)
    aqua_de = float(median_aqua_de if median_aqua_de is not None else float("inf"))
    low_fraction = float(low_pixel_fraction or 0.0)
    low_de = float(median_low_de if median_low_de is not None else float("inf"))
    weak_aqua_low_compatible = bool(
        MICRO_WEAK_AQUA_LOW_MIN <= aqua_fraction <= MICRO_WEAK_AQUA_LOW_MAX
        and aqua_de <= MICRO_WEAK_AQUA_LOW_DE_MAX
        and low_fraction >= MICRO_WEAK_AQUA_LOW_LOW_SUPPORT_MIN
        and low_de <= aqua_de + MICRO_WEAK_AQUA_LOW_MARGIN_TOL
    )
    moderate_aqua_present = bool(
        aqua_fraction >= MICRO_MODERATE_AQUA_MIN
        and aqua_de <= MICRO_MODERATE_AQUA_DE_MAX
        and low_fraction <= MICRO_MODERATE_AQUA_LOW_SUPPRESSION_MAX
    )
    very_low_moderate_aqua = bool(
        aqua_fraction >= MICRO_WEAK_AQUA_LOW_MAX
        and aqua_fraction < MICRO_MODERATE_AQUA_MIN
        and aqua_de <= MICRO_MODERATE_AQUA_DE_MAX
        and low_fraction >= MICRO_WEAK_AQUA_LOW_LOW_SUPPORT_MIN
        and low_de <= aqua_de + MICRO_WEAK_AQUA_LOW_MARGIN_TOL
    )
    if strong_aqua_confirmed:
        tier = "strong"
    elif moderate_aqua_present:
        tier = "moderate"
    elif very_low_moderate_aqua:
        tier = "very_low_moderate_low_compatible"
    elif weak_aqua_low_compatible:
        tier = "weak_low_compatible"
    else:
        tier = "none"
    return {
        "weak_aqua_low_compatible": weak_aqua_low_compatible,
        "very_low_moderate_aqua": very_low_moderate_aqua,
        "moderate_aqua_present": moderate_aqua_present,
        "strong_aqua_confirmed": bool(strong_aqua_confirmed),
        "aqua_evidence_tier": tier,
        "weak_aqua_low_range": [MICRO_WEAK_AQUA_LOW_MIN, MICRO_WEAK_AQUA_LOW_MAX],
        "moderate_aqua_threshold": MICRO_MODERATE_AQUA_MIN,
        "moderate_aqua_de_max": MICRO_MODERATE_AQUA_DE_MAX,
        "weak_aqua_low_reason": (
            "Faint aqua evidence remains compatible with low microalbumin when low-shade support is present."
            if weak_aqua_low_compatible else
            "Weak low-compatible aqua criteria not met."
        ),
        "moderate_aqua_reason": (
            "Moderate aqua evidence supports provisional 80–150 mg/L reporting when strong aqua is absent."
            if moderate_aqua_present else
            "Moderate aqua criteria not met."
        ),
        "legacy_weak_aqua_present_alias": moderate_aqua_present,
        "weak_aqua_present": moderate_aqua_present,
    }


def microalbumin_guard_from_evidence(
    *,
    current_value_float,
    low_candidate_class_mg_l=10,
    low_shade_confirmed=False,
    low_shade_confirmed_relaxed=False,
    strong_aqua_confirmed=False,
    high_value_color_verified=False,
    overbright_ood_no_chart_support=False,
    weak_aqua_low_compatible=False,
    very_low_moderate_aqua=False,
    moderate_aqua_present=False,
    allow_unconfirmed=True,
):
    """Small decision slice used by tests and by the V4 shade guard aqua-tier branches."""
    current_value_float = float(current_value_float)
    corrected_albumin_value = current_value_float
    report_mode = "exact"
    guard_applied = False
    action = "unchanged_not_evaluated"
    guard_reason = ""
    provisional_albumin_range_mg_l = None
    if current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE:
        if overbright_ood_no_chart_support:
            corrected_albumin_value = None
            report_mode = "unconfirmed"
            guard_applied = True
            action = "unconfirmed_ood_below_400"
            guard_reason = "Microalbumin estimate is below 400 mg/L, but pod colour evidence is out-of-distribution; retake required."
        elif low_shade_confirmed:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            report_mode = "exact"
            guard_applied = current_value_float != corrected_albumin_value
            action = "confirmed_low_exact_3_10_30"
            guard_reason = "Strict low-shade evidence confirmed; exact 3/10/30 mg/L selected."
        elif (low_shade_confirmed_relaxed or low_shade_confirmed) and weak_aqua_low_compatible and not strong_aqua_confirmed and not moderate_aqua_present:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            guard_applied = current_value_float != corrected_albumin_value
            report_mode = "guarded_exact"
            action = "weak_aqua_low_compatible_mapped_to_low"
            guard_reason = "Faint weak-aqua evidence was present, but low-shade evidence supported a <30 mg/L class; microalbumin was mapped to the nearest low class 3/10/30 mg/L."
        elif (low_shade_confirmed_relaxed or low_shade_confirmed) and very_low_moderate_aqua and not strong_aqua_confirmed and not moderate_aqua_present:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            guard_applied = current_value_float != corrected_albumin_value
            report_mode = "guarded_exact"
            action = "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
            guard_reason = "Very-low moderate aqua evidence was present, but low-shade evidence remained competitive; microalbumin was mapped to the nearest low class 3/10/30 mg/L."
        elif low_shade_confirmed_relaxed and not strong_aqua_confirmed:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            report_mode = "guarded_exact"
            guard_applied = current_value_float != corrected_albumin_value
            action = "relaxed_low_guard_below_400"
            guard_reason = "Relaxed low-shade evidence confirmed below 400 mg/L; weak aqua was not sufficient to override low evidence."
        elif strong_aqua_confirmed:
            corrected_albumin_value = current_value_float
            report_mode = "exact"
            guard_applied = False
            action = "strong_aqua_preserved_for_full_guard_matching"
            guard_reason = "Strong aqua evidence remains handled by the existing full guard matching branch."
        elif moderate_aqua_present and not strong_aqua_confirmed:
            corrected_albumin_value = None if allow_unconfirmed else 80.0
            provisional_albumin_range_mg_l = (80.0, 150.0)
            guard_applied = True
            report_mode = "provisional_range"
            action = "moderate_aqua_provisional_80_150"
            guard_reason = "Moderate aqua evidence is present without strong aqua; exact high value is not confirmed. Provisional 80–150 mg/L range used."
        else:
            action = "unchanged_below_400_no_guard_triggered"
            guard_reason = "Below-400 estimate retained; no low/aqua guard condition triggered."
    elif overbright_ood_no_chart_support:
        corrected_albumin_value = None
        report_mode = "unconfirmed"
        guard_applied = True
        action = "unconfirmed_high_estimate_ood"
        guard_reason = "Microalbumin estimate is >=400 mg/L, but pod colour evidence is out-of-distribution; retake required."
    elif high_value_color_verified:
        report_mode = "exact" if current_value_float < 600.0 else "high_watch"
        action = "high_value_verified_preserved"
        guard_reason = "Microalbumin estimate >=400 mg/L was preserved after high-value colour verification."
    else:
        corrected_albumin_value = None
        report_mode = "unconfirmed"
        guard_applied = True
        action = "high_value_not_verified_retest"
        guard_reason = "Microalbumin estimate >=400 mg/L was not downgraded, but high-value colour evidence was insufficient; retake required."
    return {
        "guard_applied": guard_applied,
        "corrected_albumin_value": corrected_albumin_value,
        "report_mode": report_mode,
        "provisional_albumin_range_mg_l": provisional_albumin_range_mg_l,
        "action": action,
        "guard_reason": guard_reason,
    }

def microalbumin_shade_sanity_check(
    img_rgb_uint8,
    pod_mask_bool,
    current_albumin_value,
    current_albumin_label=None,
    current_confidence=None,
    allow_unconfirmed=True,
):
    """Targeted post-hoc visual guard for microalbumin only."""
    def _normalize_confidence(conf):
        if conf is None:
            return None
        try:
            c = float(conf)
        except Exception:
            return None
        if c > 1.0:
            c = c / 100.0
        return float(np.clip(c, 0.0, 1.0))

    def _safe_float(value):
        return None if value is None else float(value)

    def _response(**kwargs):
        low_cls = int(kwargs.get("low_candidate_class_mg_l") or 10)
        low_rgb = MICROALBUMIN_LOW_EXACT_REFS.get(low_cls, MICROALBUMIN_CENTROIDS[10])
        provisional_range = kwargs.get("provisional_albumin_range_mg_l")
        if provisional_range is not None:
            provisional_range = (float(provisional_range[0]), float(provisional_range[1]))
        strong_bin = kwargs.get("strong_aqua_candidate_bin")
        return {
            "guard_name": "microalbumin_shade_sanity_check",
            "guard_applied": bool(kwargs.get("guard_applied", False)),
            "action": str(kwargs.get("action", "unchanged_not_evaluated")),
            "report_mode": str(kwargs.get("report_mode", "exact")),
            "original_albumin_value": kwargs.get("original_albumin_value"),
            "corrected_albumin_value": _safe_float(kwargs.get("corrected_albumin_value")),
            "provisional_albumin_range_mg_l": provisional_range,

            "low_candidate_class_mg_l": low_cls,
            "low_candidate_rgb": tuple(int(round(x)) for x in low_rgb),
            "low_candidate_hex": "#%02X%02X%02X" % tuple(int(round(x)) for x in low_rgb),
            "low_candidate_visual_name": MICROALBUMIN_LOW_EXACT_VISUAL_NAMES.get(low_cls, "nearest low microalbumin shade"),
            "median_de_low_by_class": {int(k): float(v) for k, v in kwargs.get("median_de_low_by_class", {}).items()},
            "median_de_3": float(kwargs.get("median_de_low_by_class", {}).get(3, 0.0)),
            "median_de_10": float(kwargs.get("median_de_low_by_class", {}).get(10, 0.0)),
            "median_de_30": float(kwargs.get("median_de_low_by_class", {}).get(30, 0.0)),
            "median_low_de": float(kwargs.get("median_low_de", 0.0)),
            "low_margin": float(kwargs.get("low_margin", 0.0)),
            "low_pixel_fraction": float(kwargs.get("low_pixel_fraction", 0.0)),
            "low_shade_confirmed": bool(kwargs.get("low_shade_confirmed", False)),
            "low_shade_confirmed_relaxed": bool(kwargs.get("low_shade_confirmed_relaxed", False)),
            "low_ambiguous": bool(kwargs.get("low_ambiguous", False)),

            "aqua_candidate_class_mg_l": int(kwargs.get("aqua_candidate_class_mg_l") or 80),
            "median_de_80": float(kwargs.get("median_de_80", 0.0)),
            "median_de_150": float(kwargs.get("median_de_150", 0.0)),
            "median_aqua_de": float(kwargs.get("median_aqua_de", 0.0)),
            "aqua_margin": float(kwargs.get("aqua_margin", 0.0)),
            "weak_aqua_present": bool(kwargs.get("moderate_aqua_present", kwargs.get("weak_aqua_present", False))),
            "weak_aqua_low_compatible": bool(kwargs.get("weak_aqua_low_compatible", False)),
            "very_low_moderate_aqua": bool(kwargs.get("very_low_moderate_aqua", False)),
            "moderate_aqua_present": bool(kwargs.get("moderate_aqua_present", kwargs.get("weak_aqua_present", False))),
            "strong_aqua_confirmed": bool(kwargs.get("strong_aqua_confirmed", False)),
            "aqua_evidence_tier": str(kwargs.get("aqua_evidence_tier", "none")),
            "weak_aqua_low_range": list(kwargs.get("weak_aqua_low_range", [MICRO_WEAK_AQUA_LOW_MIN, MICRO_WEAK_AQUA_LOW_MAX])),
            "moderate_aqua_threshold": float(kwargs.get("moderate_aqua_threshold", MICRO_MODERATE_AQUA_MIN)),
            "moderate_aqua_de_max": float(kwargs.get("moderate_aqua_de_max", MICRO_MODERATE_AQUA_DE_MAX)),
            "weak_aqua_low_reason": str(kwargs.get("weak_aqua_low_reason", "Weak low-compatible aqua criteria not met.")),
            "moderate_aqua_reason": str(kwargs.get("moderate_aqua_reason", "Moderate aqua criteria not met.")),
            "legacy_weak_aqua_present_alias": bool(kwargs.get("moderate_aqua_present", kwargs.get("weak_aqua_present", False))),
            "aqua_evidence_reporting_note": "Aqua evidence is tiered as weak-low-compatible, very-low-moderate-low-compatible, moderate, or strong. Moderate aqua replaces the previous weak-aqua-to-80/150 behavior.",
            "aqua_pixel_fraction": float(kwargs.get("aqua_pixel_fraction", 0.0)),
            "strong_aqua_candidate_bin": None if strong_bin is None else int(strong_bin),
            "strong_aqua_candidate_de": _safe_float(kwargs.get("strong_aqua_candidate_de")),
            "high_value_color_verified": bool(kwargs.get("high_value_color_verified", False)),
            "strong_aqua_confirmed_for_high": bool(kwargs.get("strong_aqua_confirmed_for_high", False)),
            "nearest_chart_bin": None if kwargs.get("nearest_chart_bin") is None else int(kwargs.get("nearest_chart_bin")),
            "nearest_chart_de": _safe_float(kwargs.get("nearest_chart_de")),
            "second_nearest_chart_de": _safe_float(kwargs.get("second_nearest_chart_de")),
            "nearest_vs_second_margin": _safe_float(kwargs.get("nearest_vs_second_margin")),
            "high_value_verification_reason": str(kwargs.get("high_value_verification_reason", "")),
            "weak_aqua_threshold_used": float(MICRO_MODERATE_AQUA_MIN),
            "moderate_aqua_threshold_used": float(MICRO_MODERATE_AQUA_MIN),
            "strong_aqua_threshold_used": float(MICRO_STRONG_AQUA_MIN),
            "strong_aqua_de_max_used": float(MICRO_STRONG_AQUA_DE_MAX),
            "high_value_confirmed_by_aqua": bool(kwargs.get("strong_aqua_confirmed_for_high", False)),
            "very_high_aqua_support": False,
            "very_high_confirmed": False,

            "value_zone": str(kwargs.get("value_zone", "below_400")),
            "overbright_not_chart_like": bool(kwargs.get("overbright_not_chart_like", False)),
            "overbright_ood_no_chart_support": bool(kwargs.get("overbright_ood_no_chart_support", False)),
            "median_L": float(kwargs.get("median_L", 0.0)),
            "median_a": float(kwargs.get("median_a", 0.0)),
            "median_b": float(kwargs.get("median_b", 0.0)),
            "median_chroma": float(kwargs.get("median_chroma", 0.0)),
            "median_rgb": tuple(int(x) for x in kwargs.get("median_rgb", (0, 0, 0))),

            "current_confidence_used": _safe_float(kwargs.get("current_confidence_used")),
            "current_is_high_watch": bool(kwargs.get("current_is_high_watch", False)),
            "current_is_very_high": bool(kwargs.get("current_is_very_high", False)),
            "low_confidence_for_very_high": bool(kwargs.get("current_is_very_high", False) and (kwargs.get("current_confidence_used") is not None) and kwargs.get("current_confidence_used") < MICRO_MIN_CONF_FOR_600),
            "allow_unconfirmed": bool(kwargs.get("allow_unconfirmed", allow_unconfirmed)),
            "confidence_bucket": str(kwargs.get("confidence_bucket", "Informational")),
            "guard_reason": str(kwargs.get("guard_reason", "")),
        }

    current_confidence_norm = _normalize_confidence(current_confidence)
    allow_unconfirmed = bool(allow_unconfirmed and MICRO_UNCONFIRMED_RETURNS_NONE)
    base = {
        "guard_applied": False,
        "action": "unchanged_not_evaluated",
        "report_mode": "exact",
        "original_albumin_value": current_albumin_value,
        "corrected_albumin_value": current_albumin_value,
        "provisional_albumin_range_mg_l": None,
        "low_candidate_class_mg_l": 10,
        "median_de_low_by_class": {3: 0.0, 10: 0.0, 30: 0.0},
        "median_low_de": 0.0,
        "low_margin": 0.0,
        "low_pixel_fraction": 0.0,
        "low_shade_confirmed": False,
        "low_shade_confirmed_relaxed": False,
        "low_ambiguous": False,
        "aqua_candidate_class_mg_l": 80,
        "median_de_80": 0.0,
        "median_de_150": 0.0,
        "median_aqua_de": 0.0,
        "aqua_margin": 0.0,
        "weak_aqua_present": False,
        "weak_aqua_low_compatible": False,
        "very_low_moderate_aqua": False,
        "moderate_aqua_present": False,
        "aqua_evidence_tier": "none",
        "strong_aqua_confirmed": False,
        "aqua_pixel_fraction": 0.0,
        "strong_aqua_candidate_bin": None,
        "strong_aqua_candidate_de": None,
        "high_value_color_verified": False,
        "strong_aqua_confirmed_for_high": False,
        "nearest_chart_bin": None,
        "nearest_chart_de": None,
        "second_nearest_chart_de": None,
        "nearest_vs_second_margin": None,
        "overbright_not_chart_like": False,
        "overbright_ood_no_chart_support": False,
        "high_value_verification_reason": "",
        "value_zone": "below_400",
        "median_L": 0.0,
        "median_a": 0.0,
        "median_b": 0.0,
        "median_chroma": 0.0,
        "median_rgb": (0, 0, 0),
        "current_confidence_used": current_confidence_norm,
        "current_is_high_watch": False,
        "current_is_very_high": False,
        "confidence_bucket": "Informational",
        "guard_reason": "",
        "allow_unconfirmed": allow_unconfirmed,
    }
    if img_rgb_uint8 is None or pod_mask_bool is None or current_albumin_value is None:
        base["action"] = "unchanged_missing_inputs"
        base["guard_reason"] = "Missing inputs for shade sanity check."
        return _response(**base)

    pod_eroded = eroded_mask(pod_mask_bool)
    total_pixels = int(pod_eroded.sum())
    if total_pixels < MICRO_SHADE_MIN_PIXELS:
        base["action"] = "unchanged_insufficient_mask"
        base["guard_reason"] = "Insufficient microalbumin pod pixels for shade sanity check."
        return _response(**base)

    rgb_pixels = np.asarray(img_rgb_uint8, dtype=np.uint8)[pod_eroded]
    lab_pixels = rgb2lab(rgb_pixels.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float64)
    median_lab = np.median(lab_pixels, axis=0)
    median_L = float(median_lab[0])
    median_a = float(median_lab[1])
    median_b = float(median_lab[2])
    median_chroma = float(np.sqrt(median_a * median_a + median_b * median_b))
    median_rgb = tuple(int(round(x)) for x in np.median(rgb_pixels, axis=0))
    base.update({
        "median_L": median_L,
        "median_a": median_a,
        "median_b": median_b,
        "median_chroma": median_chroma,
        "median_rgb": median_rgb,
    })

    de_low_by_class = {
        cls: _deltae_pixels_to_lab_ref(lab_pixels, lab_ref)
        for cls, lab_ref in MICROALBUMIN_LOW_EXACT_REFS_LAB.items()
    }
    median_de_low_by_class = {
        cls: float(np.median(de_vals))
        for cls, de_vals in de_low_by_class.items()
    }
    low_sorted = sorted(median_de_low_by_class.items(), key=lambda item: item[1])
    low_candidate_class_mg_l = int(low_sorted[0][0])
    median_low_de = float(low_sorted[0][1])
    second_low_de = float(low_sorted[1][1]) if len(low_sorted) > 1 else float("inf")
    low_margin = float(second_low_de - median_low_de)
    low_pixel_de = np.minimum.reduce([de_low_by_class[3], de_low_by_class[10], de_low_by_class[30]])
    low_pixel_fraction = float(np.mean(low_pixel_de <= MICRO_LOW_DE_MAX))
    low_shade_confirmed = bool(
        median_low_de <= MICRO_LOW_DE_MAX
        and low_pixel_fraction >= MICRO_LOW_PIXEL_FRACTION_MIN
    )
    low_ambiguous = bool(low_shade_confirmed and low_margin < MICRO_LOW_MARGIN_AMBIGUOUS)

    de_80 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_AQUA_CONFIRM_REFS_LAB[80])
    de_150 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_AQUA_CONFIRM_REFS_LAB[150])
    # 80/150 remain the named aqua comparators. Controlled strong-aqua
    # candidates are also included as aqua-like evidence so centroids from
    # 250/400/600 can unlock the guarded 150+ matching path.
    aqua_evidence_arrays = [de_80, de_150]
    aqua_evidence_medians = {80: float(np.median(de_80)), 150: float(np.median(de_150))}
    for lbl, lab_ref in MICRO_STRONG_AQUA_ALLOWED_REFS_LAB.items():
        if lbl in (80, 150):
            continue
        de_allowed = _deltae_pixels_to_lab_ref(lab_pixels, lab_ref)
        aqua_evidence_arrays.append(de_allowed)
        aqua_evidence_medians[int(lbl)] = float(np.median(de_allowed))
    aqua_pixel_de = np.minimum.reduce(aqua_evidence_arrays)
    median_de_80 = aqua_evidence_medians[80]
    median_de_150 = aqua_evidence_medians[150]
    aqua_candidate_class_mg_l = 80 if median_de_80 <= median_de_150 else 150
    median_aqua_de = float(min(aqua_evidence_medians.values()))
    aqua_margin = float(abs(median_de_80 - median_de_150))
    aqua_pixel_fraction = float(np.mean(
        (aqua_pixel_de <= MICRO_STRONG_AQUA_DE_MAX)
        & (aqua_pixel_de + MICRO_AQUA_ADVANTAGE_MARGIN < low_pixel_de)
    ))
    current_value_float = float(current_albumin_value)
    value_zone = "below_400" if current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE else "above_or_equal_400"
    current_is_low = current_value_float <= 30.0
    current_is_mid = 30.0 < current_value_float < MICRO_HIGH_WATCH_MIN
    current_is_high_watch = MICRO_HIGH_WATCH_MIN <= current_value_float < MICRO_VERY_HIGH_GUARD_MIN
    current_is_very_high = current_value_float >= MICRO_VERY_HIGH_GUARD_MIN

    strong_aqua_confirmed = bool(
        current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE
        and aqua_pixel_fraction >= MICRO_STRONG_AQUA_MIN
        and median_aqua_de <= MICRO_STRONG_AQUA_DE_MAX
        and median_aqua_de + MICRO_AQUA_ADVANTAGE_MARGIN < median_low_de
    )
    aqua_tiers = evaluate_microalbumin_aqua_tiers(
        aqua_pixel_fraction=aqua_pixel_fraction,
        median_aqua_de=median_aqua_de,
        low_pixel_fraction=low_pixel_fraction,
        median_low_de=median_low_de,
        strong_aqua_confirmed=strong_aqua_confirmed,
    )
    weak_aqua_low_compatible = aqua_tiers["weak_aqua_low_compatible"]
    very_low_moderate_aqua = aqua_tiers["very_low_moderate_aqua"]
    moderate_aqua_present = bool(
        current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE
        and aqua_tiers["moderate_aqua_present"]
    )
    weak_aqua_present = moderate_aqua_present
    aqua_tiers["moderate_aqua_present"] = moderate_aqua_present
    aqua_tiers["legacy_weak_aqua_present_alias"] = moderate_aqua_present
    aqua_tiers["weak_aqua_present"] = moderate_aqua_present
    overbright_not_chart_like = bool(
        median_L >= MICRO_OVERBRIGHT_L_MAX
        and not low_shade_confirmed
        and not strong_aqua_confirmed
    )
    chart_like_support = bool(
        median_low_de <= MICRO_CHARTLIKE_DE_MAX
        or median_aqua_de <= MICRO_CHARTLIKE_DE_MAX
    )
    low_or_aqua_pixel_support = bool(
        low_pixel_fraction > MICRO_OOD_PIXEL_FRACTION_MAX
        or aqua_pixel_fraction > MICRO_OOD_PIXEL_FRACTION_MAX
    )
    overbright_ood_no_chart_support = bool(
        overbright_not_chart_like
        and not chart_like_support
        and not low_or_aqua_pixel_support
    )
    low_shade_confirmed_relaxed = bool(
        current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE
        and not overbright_ood_no_chart_support
        and median_low_de <= MICRO_LOW_DE_RELAXED_MAX
        and low_pixel_fraction >= MICRO_LOW_PIXEL_FRACTION_RELAXED_MIN
        and median_low_de + MICRO_LOW_VS_AQUA_MARGIN_MIN <= median_aqua_de
    )
    strong_aqua_confirmed_for_high = bool(
        current_value_float >= MICRO_HIGH_VERIFY_MIN_VALUE
        and aqua_pixel_fraction >= MICRO_HIGH_VERIFY_MIN_AQUA_FRACTION
        and median_aqua_de <= MICRO_STRONG_AQUA_DE_MAX
        and median_aqua_de + MICRO_AQUA_ADVANTAGE_MARGIN < median_low_de
    )
    nearest_chart_bin, nearest_chart_de, second_nearest_chart_de, nearest_vs_second_margin = _nearest_micro_chart_bin_with_margin(median_lab)
    nearest_high_bin_verified = bool(
        current_value_float >= MICRO_HIGH_VERIFY_MIN_VALUE
        and nearest_chart_bin in [400, 600, 800, 1000, 1400]
        and nearest_chart_de is not None
        and nearest_chart_de <= MICRO_HIGH_VERIFY_DE_MAX
        and nearest_vs_second_margin is not None
        and nearest_vs_second_margin >= MICRO_HIGH_VERIFY_MARGIN_MIN
    )
    high_value_color_verified = bool(strong_aqua_confirmed_for_high or nearest_high_bin_verified)
    if current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE:
        high_value_verification_reason = "not_applicable_below_400"
    elif strong_aqua_confirmed_for_high and nearest_high_bin_verified:
        high_value_verification_reason = "strong_aqua_and_nearest_high_bin_verified"
    elif strong_aqua_confirmed_for_high:
        high_value_verification_reason = "strong_aqua_verified_high"
    elif nearest_high_bin_verified:
        high_value_verification_reason = "nearest_high_chart_bin_verified"
    else:
        high_value_verification_reason = "insufficient_high_value_colour_evidence"

    if strong_aqua_confirmed:
        strong_aqua_candidate_bin, strong_aqua_candidate_de = _nearest_micro_allowed_bin_lab(
            median_lab,
            [150, 250, 400],
        )
    else:
        strong_aqua_candidate_bin = None
        strong_aqua_candidate_de = None

    base.update({
        "low_candidate_class_mg_l": low_candidate_class_mg_l,
        "median_de_low_by_class": median_de_low_by_class,
        "median_low_de": median_low_de,
        "low_margin": low_margin,
        "low_pixel_fraction": low_pixel_fraction,
        "low_shade_confirmed": low_shade_confirmed,
        "low_shade_confirmed_relaxed": low_shade_confirmed_relaxed,
        "low_ambiguous": low_ambiguous,
        "aqua_candidate_class_mg_l": aqua_candidate_class_mg_l,
        "median_de_80": median_de_80,
        "median_de_150": median_de_150,
        "median_aqua_de": median_aqua_de,
        "aqua_margin": aqua_margin,
        **aqua_tiers,
        "weak_aqua_present": weak_aqua_present,
        "strong_aqua_confirmed": strong_aqua_confirmed,
        "aqua_pixel_fraction": aqua_pixel_fraction,
        "strong_aqua_candidate_bin": strong_aqua_candidate_bin,
        "strong_aqua_candidate_de": None if strong_aqua_candidate_de is None else float(strong_aqua_candidate_de),
        "high_value_color_verified": high_value_color_verified,
        "strong_aqua_confirmed_for_high": strong_aqua_confirmed_for_high,
        "nearest_chart_bin": nearest_chart_bin,
        "nearest_chart_de": nearest_chart_de,
        "second_nearest_chart_de": second_nearest_chart_de,
        "nearest_vs_second_margin": nearest_vs_second_margin,
        "overbright_not_chart_like": overbright_not_chart_like,
        "overbright_ood_no_chart_support": overbright_ood_no_chart_support,
        "high_value_verification_reason": high_value_verification_reason,
        "value_zone": value_zone,
        "current_is_high_watch": current_is_high_watch,
        "current_is_very_high": current_is_very_high,
    })

    guard_applied = False
    corrected_albumin_value = current_value_float
    report_mode = "exact"
    provisional_albumin_range_mg_l = None
    action = "unchanged_not_evaluated"
    guard_reason = ""

    if current_value_float < MICRO_HIGH_VERIFY_MIN_VALUE:
        if overbright_ood_no_chart_support:
            corrected_albumin_value = None
            report_mode = "unconfirmed"
            guard_applied = True
            action = "unconfirmed_ood_below_400"
            guard_reason = (
                "Microalbumin estimate is below 400 mg/L, but pod colour evidence is out-of-distribution; retake required."
            )
        elif low_shade_confirmed:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            report_mode = "exact"
            guard_applied = current_value_float != corrected_albumin_value
            action = "confirmed_low_exact_3_10_30"
            guard_reason = (
                "Strict low-shade evidence confirmed; exact 3/10/30 mg/L selected."
            )
        elif (low_shade_confirmed_relaxed or low_shade_confirmed) and weak_aqua_low_compatible and not strong_aqua_confirmed and not moderate_aqua_present:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            guard_applied = current_value_float != corrected_albumin_value
            report_mode = "guarded_exact"
            action = "weak_aqua_low_compatible_mapped_to_low"
            guard_reason = (
                "Faint weak-aqua evidence was present, but low-shade evidence supported a <30 mg/L class; "
                "microalbumin was mapped to the nearest low class 3/10/30 mg/L."
            )
        elif (low_shade_confirmed_relaxed or low_shade_confirmed) and very_low_moderate_aqua and not strong_aqua_confirmed and not moderate_aqua_present:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            guard_applied = current_value_float != corrected_albumin_value
            report_mode = "guarded_exact"
            action = "very_low_moderate_aqua_with_low_evidence_mapped_to_low"
            guard_reason = (
                "Very-low moderate aqua evidence was present, but low-shade evidence remained competitive; "
                "microalbumin was mapped to the nearest low class 3/10/30 mg/L."
            )
        elif low_shade_confirmed_relaxed and not strong_aqua_confirmed:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            report_mode = "guarded_exact"
            guard_applied = current_value_float != corrected_albumin_value
            action = "relaxed_low_guard_below_400"
            guard_reason = (
                "Relaxed low-shade evidence confirmed below 400 mg/L; weak aqua was not sufficient to override low evidence."
            )
        elif strong_aqua_confirmed:
            candidate, candidate_de = _nearest_micro_allowed_bin_lab(
                median_lab,
                [150, 250, 400],
            )
            corrected_albumin_value = float(candidate)
            report_mode = "exact"
            guard_applied = current_value_float != corrected_albumin_value
            action = f"strong_aqua_below_400_matched_{candidate}"
            guard_reason = (
                f"Strong aqua evidence confirmed below 400 mg/L; nearest allowed bin selected as {candidate} mg/L."
            )
        elif moderate_aqua_present and not strong_aqua_confirmed:
            corrected_albumin_value = None if allow_unconfirmed else 80.0
            provisional_albumin_range_mg_l = (80.0, 150.0)
            report_mode = "provisional_range"
            guard_applied = True
            action = "moderate_aqua_provisional_80_150"
            guard_reason = (
                "Moderate aqua evidence is present without strong aqua; exact high value is not confirmed. "
                "Provisional 80–150 mg/L range used."
            )
        else:
            corrected_albumin_value = current_value_float
            report_mode = "exact"
            guard_applied = False
            action = "unchanged_below_400_no_guard_triggered"
            guard_reason = (
                "Below-400 estimate retained; no low/aqua guard condition triggered."
            )
    else:
        if overbright_ood_no_chart_support:
            corrected_albumin_value = None
            report_mode = "unconfirmed"
            guard_applied = True
            action = "unconfirmed_high_estimate_ood"
            guard_reason = (
                "Microalbumin estimate is >=400 mg/L, but pod colour evidence is out-of-distribution; retake required."
            )
        elif high_value_color_verified:
            corrected_albumin_value = current_value_float
            report_mode = "exact" if current_value_float < 600.0 else "high_watch"
            guard_applied = False
            action = "high_value_verified_preserved"
            guard_reason = (
                "Microalbumin estimate >=400 mg/L was preserved after high-value colour verification."
            )
        else:
            corrected_albumin_value = None
            report_mode = "unconfirmed"
            guard_applied = True
            action = "high_value_not_verified_retest"
            guard_reason = (
                "Microalbumin estimate >=400 mg/L was not downgraded, but high-value colour evidence was insufficient; retake required."
            )

    if low_ambiguous and low_shade_confirmed:
        guard_reason += " Low classes 3/10/30 are close; nearest low class selected by median ΔE."

    if report_mode == "exact" and low_shade_confirmed:
        confidence_bucket = "Moderate" if low_ambiguous else "High"
    elif report_mode == "exact" and strong_aqua_confirmed:
        confidence_bucket = "Moderate"
    elif report_mode == "guarded_exact":
        confidence_bucket = "Moderate"
    elif report_mode == "provisional_range":
        confidence_bucket = "Low"
    elif report_mode == "high_watch":
        confidence_bucket = "Low"
    elif report_mode == "unconfirmed":
        confidence_bucket = "Low"
    else:
        confidence_bucket = "Informational"

    base.update({
        "guard_applied": guard_applied,
        "corrected_albumin_value": corrected_albumin_value,
        "report_mode": report_mode,
        "provisional_albumin_range_mg_l": provisional_albumin_range_mg_l,
        "action": action,
        "guard_reason": guard_reason,
        "confidence_bucket": confidence_bucket,
    })

    if report_mode in ("unconfirmed", "provisional_range") or guard_applied:
        logging.warning(
            "Microalbumin shade guard action=%s original=%s corrected=%s report_mode=%s low_fraction=%.3f aqua_fraction=%.3f confidence=%s reason=%s",
            action,
            current_albumin_value,
            corrected_albumin_value,
            report_mode,
            low_pixel_fraction,
            aqua_pixel_fraction,
            current_confidence_norm,
            guard_reason,
        )
    return _response(**base)



def _valid_positive_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _valid_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def convert_creatinine_to_si(creatinine_mg_dl):
    cr = _valid_positive_float(creatinine_mg_dl)
    if cr is None:
        return None
    umol_l = cr * 88.4
    mmol_l = cr * 0.0884
    return {
        "creatinine_mg_dl": round(cr, 2),
        "creatinine_umol_l": round(umol_l, 2),
        "creatinine_mmol_l": round(mmol_l, 4),
        "creatinine_si_display": f"{round(umol_l, 2):.2f} µmol/L",
    }


def convert_microalbumin_to_si(albumin_mg_l):
    albumin = _valid_float(albumin_mg_l)
    if albumin is None:
        return None
    g_l = albumin / 1000.0
    return {
        "microalbumin_mg_l": round(albumin, 2),
        "microalbumin_g_l": round(g_l, 3),
        "microalbumin_ug_ml": round(albumin, 2),
        "microalbumin_si_display": f"{g_l:.3f} g/L",
        "microalbumin_equivalent_display": f"{round(albumin, 2):.2f} µg/mL",
    }


def stage_acr_si_mg_mmol(acr_mg_mmol):
    acr = _valid_float(acr_mg_mmol)
    if acr is None:
        return {
            "acr_si_stage": "Unconfirmed / retest",
            "acr_si_stage_code": "unconfirmed",
            "acr_si_reference_range": None,
        }
    if acr < 3:
        return {
            "acr_si_stage": "A1: Normal to mildly increased",
            "acr_si_stage_code": "A1",
            "acr_si_reference_range": "< 3 mg/mmol",
        }
    if acr <= 30:
        return {
            "acr_si_stage": "A2: Moderately increased",
            "acr_si_stage_code": "A2",
            "acr_si_reference_range": "3–30 mg/mmol",
        }
    return {
        "acr_si_stage": "A3: Severely increased",
        "acr_si_stage_code": "A3",
        "acr_si_reference_range": "> 30 mg/mmol",
    }


def calculate_acr_si(albumin_mg_l, creatinine_mg_dl):
    albumin = _valid_float(albumin_mg_l)
    cr = _valid_positive_float(creatinine_mg_dl)
    if albumin is None or cr is None:
        staged = stage_acr_si_mg_mmol(None)
        return {"acr_mg_mmol": None, "acr_si_display": "Unconfirmed / retest", **staged}
    acr = albumin / (cr * 0.0884)
    acr_rounded = round(acr, 2)
    return {
        "acr_mg_mmol": acr_rounded,
        "acr_si_display": f"{acr_rounded:.2f} mg/mmol",
        **stage_acr_si_mg_mmol(acr),
    }


def calculate_acr_si_range(albumin_range_mg_l, creatinine_mg_dl):
    cr = _valid_positive_float(creatinine_mg_dl)
    if albumin_range_mg_l is None or cr is None:
        return None
    try:
        low_albumin, high_albumin = albumin_range_mg_l
        low_albumin = float(low_albumin)
        high_albumin = float(high_albumin)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low_albumin) and math.isfinite(high_albumin)):
        return None
    acr_low = low_albumin / (cr * 0.0884)
    acr_high = high_albumin / (cr * 0.0884)
    low = round(acr_low, 2)
    high = round(acr_high, 2)
    if acr_high < 3:
        stage = "Provisional A1 / retest"
        code = "A1_provisional"
    elif acr_low >= 3 and acr_high <= 30:
        stage = "Provisional A2 / retest"
        code = "A2_provisional"
    elif acr_low < 3 and acr_high <= 30:
        stage = "Provisional A1/A2 boundary / retest"
        code = "A1_A2_boundary_provisional"
    elif acr_low <= 30 and acr_high > 30:
        stage = "A2/A3 boundary unconfirmed / retest"
        code = "A2_A3_boundary_unconfirmed"
    else:
        stage = "Unconfirmed / retest"
        code = "unconfirmed"
    return {
        "acr_si_range_mg_mmol": (low, high),
        "acr_si_range_display": f"{low:.2f}–{high:.2f} mg/mmol",
        "acr_si_stage": stage,
        "acr_si_stage_code": code,
    }


def _unconfirmed_acr_si():
    return {"acr_mg_mmol": None, "acr_si_display": "Unconfirmed / retest", **stage_acr_si_mg_mmol(None)}

def calculate_uacr_range_and_stage(albumin_range_mg_l, creatinine_mg_dl):
    if albumin_range_mg_l is None or creatinine_mg_dl is None or creatinine_mg_dl <= 0:
        return None

    low_albumin, high_albumin = albumin_range_mg_l
    uacr_low = 100.0 * float(low_albumin) / float(creatinine_mg_dl)
    uacr_high = 100.0 * float(high_albumin) / float(creatinine_mg_dl)

    if uacr_high < 30:
        stage = "Provisional A1 / retest"
        code = "A1_provisional"
    elif uacr_low >= 30 and uacr_high <= 300:
        stage = "Provisional A2 / retest"
        code = "A2_provisional"
    elif uacr_low < 30 and uacr_high <= 300:
        stage = "Provisional A1/A2 boundary / retest"
        code = "A1_A2_boundary_provisional"
    elif uacr_low <= 300 and uacr_high > 300:
        stage = "A2/A3 boundary unconfirmed / retest"
        code = "A2_A3_boundary_unconfirmed"
    else:
        stage = "Unconfirmed / retake image"
        code = "unconfirmed"

    return {
        "uacr_range_mg_g": (round(uacr_low, 2), round(uacr_high, 2)),
        "uacr_display": f"{round(uacr_low, 2):.2f}–{round(uacr_high, 2):.2f} mg/g",
        "uacr_stage": stage,
        "uacr_stage_code": code,
    }

def derive_microalbumin_guarded_uacr_scenario(
    shade_guard: dict,
    creatinine_mg_dl,
    original_albumin_value=None,
    exact_albumin_value=None,
):
    """Derive conservative provisional UACR evidence from microalbumin shade guard."""
    unavailable = {
        "provisional_available": False,
        "provisional_guard_scenario": "no_safe_a1_a2_mapping",
        "provisional_reason": "Insufficient shade evidence for provisional A1/A2 mapping.",
        "non_a3_supported": False,
    }
    if not shade_guard or creatinine_mg_dl is None:
        return unavailable
    try:
        cr = float(creatinine_mg_dl)
    except Exception:
        return unavailable
    if cr <= 0:
        return unavailable

    action = str(shade_guard.get("action", ""))
    report_mode = str(shade_guard.get("report_mode", ""))
    low_visual = shade_guard.get("low_candidate_visual_name")
    median_L = shade_guard.get("median_L")
    low_fraction = shade_guard.get("low_pixel_fraction")
    aqua_fraction = shade_guard.get("aqua_pixel_fraction")
    weak_aqua = bool(shade_guard.get("weak_aqua_present", False))
    strong_aqua = bool(shade_guard.get("strong_aqua_confirmed", False))
    overbright = bool(shade_guard.get("overbright_not_chart_like", False))
    low_confirmed = bool(shade_guard.get("low_shade_confirmed", False))

    albumin_range = shade_guard.get("provisional_albumin_range_mg_l")
    if report_mode != "provisional_range" or albumin_range is None:
        return unavailable

    range_result = calculate_uacr_range_and_stage(albumin_range, cr)
    if range_result is None:
        return unavailable

    albumin_range = (float(albumin_range[0]), float(albumin_range[1]))
    return {
        "provisional_available": True,
        "report_mode": "provisional_range",
        "provisional_guard_scenario": action or "microalbumin_provisional_range",
        "provisional_reason": shade_guard.get("guard_reason", "Microalbumin shade evidence supports provisional reporting only."),
        "provisional_albumin_range_mg_l": albumin_range,
        "provisional_albumin_display": f"{albumin_range[0]:.0f}–{albumin_range[1]:.0f} mg/L",
        "creatinine_used_mg_dl": float(cr),
        "provisional_uacr_range_mg_g": range_result["uacr_range_mg_g"],
        "provisional_uacr_display": range_result["uacr_display"],
        "provisional_uacr_stage": range_result["uacr_stage"],
        "provisional_uacr_stage_code": range_result["uacr_stage_code"],
        "non_a3_supported": bool(range_result["uacr_range_mg_g"][1] <= 300.0),
        "source_guard_action": action,
        "source_low_visual": None if low_visual is None else str(low_visual),
        "source_median_L": None if median_L is None else float(median_L),
        "source_low_pixel_fraction": None if low_fraction is None else float(low_fraction),
        "source_aqua_pixel_fraction": None if aqua_fraction is None else float(aqua_fraction),
        "source_weak_aqua": weak_aqua,
        "source_strong_aqua": strong_aqua,
        "source_overbright": overbright,
        "source_low_confirmed": low_confirmed,
        "source_original_albumin_value": original_albumin_value,
        "source_exact_albumin_value": exact_albumin_value,
    }

def choose_microalbumin_report_display(exact_value, shade_guard, guarded_scenario):
    """Decide what to display for microalbumin under shade-guarded reporting."""
    shade_guard = shade_guard or {}
    action = str(shade_guard.get("action", ""))
    report_mode = str(shade_guard.get("report_mode", "exact"))
    scenario_available = bool(isinstance(guarded_scenario, dict) and guarded_scenario.get("provisional_available"))
    range_tuple = shade_guard.get("provisional_albumin_range_mg_l")
    if range_tuple is None and scenario_available:
        range_tuple = guarded_scenario.get("provisional_albumin_range_mg_l")
    range_display = None
    if range_tuple is not None:
        range_tuple = (float(range_tuple[0]), float(range_tuple[1]))
        range_display = f"{range_tuple[0]:.0f}–{range_tuple[1]:.0f} mg/L"

    def _exact(prefix=""):
        text = f"{prefix}{float(exact_value):.0f} mg/L"
        if action.startswith("strong_aqua_matched_") or action.startswith("strong_aqua_below_400_matched_"):
            text += ", aqua-confirmed"
        return {
            "microalbumin_report_mode": report_mode if report_mode in ("guarded_exact", "high_watch") else "exact",
            "microalbumin_display_text": text,
            "microalbumin_exact_value_mg_l": float(exact_value),
            "microalbumin_range_mg_l": None,
            "microalbumin_range_display": None,
            "retest_recommended": report_mode == "high_watch",
        }

    if report_mode == "provisional_range":
        return {
            "microalbumin_report_mode": "provisional_range",
            "microalbumin_display_text": f"Provisional {range_display or 'range unavailable'} / retest",
            "microalbumin_exact_value_mg_l": None,
            "microalbumin_range_mg_l": tuple(range_tuple) if range_tuple is not None else None,
            "microalbumin_range_display": range_display,
            "retest_recommended": True,
        }
    if report_mode == "unconfirmed" or exact_value is None:
        return {
            "microalbumin_report_mode": "unconfirmed",
            "microalbumin_display_text": "Unconfirmed / retake image",
            "microalbumin_exact_value_mg_l": None,
            "microalbumin_range_mg_l": None,
            "microalbumin_range_display": None,
            "retest_recommended": True,
        }
    if report_mode == "guarded_exact":
        return _exact(prefix="Guarded ")
    if report_mode == "high_watch":
        return {
            "microalbumin_report_mode": "high_watch",
            "microalbumin_display_text": f"High-watch {float(exact_value):.0f} mg/L / retest recommended",
            "microalbumin_exact_value_mg_l": float(exact_value),
            "microalbumin_range_mg_l": None,
            "microalbumin_range_display": None,
            "retest_recommended": True,
        }
    return _exact()

# ───────────────────────────────
# White balance & robust color extraction
# ───────────────────────────────

def gray_world_white_balance(img_uint8):
    img = img_uint8.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + WB_EPS
    gray = float(means.mean())
    gains = gray / means
    balanced = np.clip(img * gains, 0, 255).astype(np.uint8)
    return balanced

def eroded_mask(mask_bool, ksize=POD_ERODE_KERNEL, iterations=1):
    if not mask_bool.any(): return mask_bool
    k = np.ones((ksize, ksize), np.uint8)
    er = cv2.erode(mask_bool.astype(np.uint8), k, iterations=iterations)
    return er.astype(bool)

def masked_median_rgb(img_uint8, mask_bool):
    if not mask_bool.any(): return np.array([0, 0, 0], dtype=np.uint8)
    pix = img_uint8[mask_bool]
    med = np.median(pix, axis=0).round().astype(np.uint8)
    return med

def _clamp(val, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(val)))

def _compute_mask_quality(mask_bool, image_shape_hw, thresholds):
    h, w = image_shape_hw
    pod_area = int(mask_bool.sum())
    total_area = float(h * w)
    area_ratio = (pod_area / total_area) if total_area > 0 else 0.0

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return {
            'mask_area_ratio': 0.0,
            'edge_touch_ratio': 1.0,
            'eccentricity': 1.0,
            'flag_mask_quality': 1,
            'mask_quality_reasons': ['empty_mask'],
        }

    edge_hits = int(((ys == 0) | (ys == h - 1) | (xs == 0) | (xs == w - 1)).sum())
    edge_touch_ratio = edge_hits / max(pod_area, 1)

    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eccentricity = 0.0
    if contours and len(contours[0]) >= 5:
        (_, _), (maj, minr), _ = cv2.fitEllipse(max(contours, key=cv2.contourArea))
        major = max(maj, minr)
        minor = max(min(maj, minr), 1e-6)
        eccentricity = np.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))

    reasons = []
    if area_ratio < thresholds['mask_area_min']:
        reasons.append('mask_too_small')
    if area_ratio > thresholds['mask_area_max']:
        reasons.append('mask_too_large')
    if edge_touch_ratio > thresholds['edge_touch_ratio']:
        reasons.append('mask_touches_border')
    if eccentricity > thresholds['eccentricity_max']:
        reasons.append('mask_extreme_eccentricity')

    return {
        'mask_area_ratio': round(area_ratio, 6),
        'edge_touch_ratio': round(edge_touch_ratio, 6),
        'eccentricity': round(float(eccentricity), 6),
        'flag_mask_quality': 1 if reasons else 0,
        'mask_quality_reasons': reasons,
    }

def _compute_quality_metrics(img_rgb_uint8, mask_bool):
    if not mask_bool.any():
        return {
            'sigma_L': 0.0,
            'mad_ab': 0.0,
            'de_p90': 0.0,
            'texture_score': 0.0,
            'glare_area_ratio': 0.0,
            'glare_cluster_max_ratio': 0.0,
            'white_spot_count': 0,
            'glare_pixels': 0,
        }

    lab = rgb2lab(img_rgb_uint8)
    l_channel = lab[..., 0]
    a_channel = lab[..., 1]
    b_channel = lab[..., 2]

    l_vals = l_channel[mask_bool]
    a_vals = a_channel[mask_bool]
    b_vals = b_channel[mask_bool]
    chroma_vals = np.sqrt(a_vals * a_vals + b_vals * b_vals)

    sigma_l = float(np.std(l_vals))
    mad_ab = float(np.median(np.abs(chroma_vals - np.median(chroma_vals))))

    med_l, med_a, med_b = np.median(l_vals), np.median(a_vals), np.median(b_vals)
    de_vals = np.sqrt((l_vals - med_l) ** 2 + (a_vals - med_a) ** 2 + (b_vals - med_b) ** 2)
    de_p90 = float(np.percentile(de_vals, 90))

    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    local_mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=1.2)
    local_sq_mean = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), sigmaX=1.2)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    texture_score = float(np.median(local_std[mask_bool]))

    hsv = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    _ = h_ch  # Explicitly unused
    glare_candidates = (
        ((v_ch >= QUALITY_THRESHOLDS['glare_v_high']) & (s_ch <= QUALITY_THRESHOLDS['glare_s_low'])) |
        (l_channel >= QUALITY_THRESHOLDS['glare_l_high'])
    ) & mask_bool

    glare_u8 = glare_candidates.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glare_u8, connectivity=8)
    min_blob_area = 3
    glare_pixels = 0
    spot_count = 0
    largest_blob = 0
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_blob_area:
            glare_pixels += area
            spot_count += 1
            largest_blob = max(largest_blob, area)

    pod_pixels = int(mask_bool.sum())
    glare_area_ratio = glare_pixels / max(pod_pixels, 1)
    glare_cluster_ratio = largest_blob / max(pod_pixels, 1)

    return {
        'sigma_L': round(sigma_l, 4),
        'mad_ab': round(mad_ab, 4),
        'de_p90': round(de_p90, 4),
        'texture_score': round(texture_score, 4),
        'glare_area_ratio': round(float(glare_area_ratio), 6),
        'glare_cluster_max_ratio': round(float(glare_cluster_ratio), 6),
        'white_spot_count': int(spot_count),
        'glare_pixels': int(glare_pixels),
    }

def _derive_quality_flags(metrics, mask_quality, thresholds):
    non_uniform_votes = 0
    non_uniform_votes += int(metrics['sigma_L'] > thresholds['sigma_l'])
    non_uniform_votes += int(metrics['mad_ab'] > thresholds['mad_ab'])
    non_uniform_votes += int(metrics['de_p90'] > thresholds['de_p90'])
    non_uniform_votes += int(metrics['texture_score'] > thresholds['texture_score'])

    flag_non_uniform = 1 if non_uniform_votes >= 2 else 0
    flag_glare = 1 if (
        metrics['glare_area_ratio'] > thresholds['glare_area_ratio'] or
        metrics['glare_cluster_max_ratio'] > thresholds['glare_cluster_ratio'] or
        metrics['white_spot_count'] > thresholds['glare_spot_count']
    ) else 0
    flag_mask_quality = int(mask_quality['flag_mask_quality'])

    return {
        'flag_non_uniform': flag_non_uniform,
        'flag_glare': flag_glare,
        'flag_mask_quality': flag_mask_quality,
    }

def _confidence_from_quality(metrics, flags):
    p_non_uniform = _clamp(
        (metrics['de_p90'] - SOFT_HARD_THRESHOLDS['de_soft']) /
        max(SOFT_HARD_THRESHOLDS['de_hard'] - SOFT_HARD_THRESHOLDS['de_soft'], 1e-6)
    )
    p_glare = _clamp(
        (metrics['glare_area_ratio'] - SOFT_HARD_THRESHOLDS['ga_soft']) /
        max(SOFT_HARD_THRESHOLDS['ga_hard'] - SOFT_HARD_THRESHOLDS['ga_soft'], 1e-6)
    )
    p_mask = 1.0 if flags['flag_mask_quality'] else 0.0

    penalty = (
        CONFIDENCE_WEIGHTS['non_uniform'] * p_non_uniform +
        CONFIDENCE_WEIGHTS['glare'] * p_glare +
        CONFIDENCE_WEIGHTS['mask'] * p_mask
    )
    confidence = _clamp(1.0 - penalty)

    if confidence >= 0.85:
        bucket = 'High'
    elif confidence >= 0.65:
        bucket = 'Moderate'
    else:
        bucket = 'Low'

    return {
        'confidence': round(confidence, 4),
        'confidence_pct': round(confidence * 100.0, 1),
        'confidence_bucket': bucket,
        'penalties': {
            'p_non_uniform': round(p_non_uniform, 4),
            'p_glare': round(p_glare, 4),
            'p_mask': round(p_mask, 4),
        }
    }

def _estimate_from_chart(pod_type, rgb_obs):
    lab_obs = _rgb_to_lab_triplet(tuple(map(int, rgb_obs)))
    if pod_type == 'creatinine':
        label, _ = nearest_creatinine_centroid_lab(lab_obs)
        return _creatinine_label_to_value(label), label
    label, _ = nearest_micro_centroid_lab(lab_obs)
    return float(label) if label is not None else None, label

def _snap_with_hysteresis(value, confidence, pod_type):
    if value is None:
        return None, None, None
    refs = sorted(REFERENCE_VALUES[pod_type].values())
    nearest = min(refs, key=lambda v: abs(v - value))
    boundary_dist = min([abs(value - b) for b in refs]) if refs else float('inf')
    snapped = nearest
    if confidence < 0.65 and boundary_dist < HYSTERESIS_BAND[pod_type]:
        # Safer behavior under low confidence near boundaries: avoid upward jump.
        lower_or_equal = [v for v in refs if v <= value]
        snapped = max(lower_or_equal) if lower_or_equal else nearest
    return snapped, boundary_dist, nearest

def _apply_quality_corrections(pod_type, v_reg, v_color, conf_data, flags, metrics):
    conf = conf_data['confidence']
    v_reg_use = v_reg if v_reg is not None else v_color
    v_color_use = v_color if v_color is not None else v_reg_use
    if v_reg_use is None:
        return {
            'raw_regression': None,
            'raw_color_chart': None,
            'fused': None,
            'corrected': None,
            'snapped': None,
            'alpha': 0.0,
            'correction_reason': 'No value available for correction',
            'boundary_distance': None,
        }

    v_fused = conf * float(v_reg_use) + (1.0 - conf) * float(v_color_use)

    severity = 0.0
    if flags['flag_non_uniform']:
        severity += 0.5
    if flags['flag_glare']:
        severity += 0.5
    severity += 0.3 * conf_data['penalties']['p_mask']
    severity = _clamp(severity, 0.0, 1.0)
    alpha = 0.08 * severity

    v_corrected = v_fused * (1.0 - alpha)
    snapped, boundary_dist, _ = _snap_with_hysteresis(v_corrected, conf, 'pod1' if pod_type == 'creatinine' else 'pod2')

    reasons = []
    if flags['flag_non_uniform']:
        reasons.append(f"non-uniformity (de_p90={metrics['de_p90']})")
    if flags['flag_glare']:
        reasons.append(f"glare ratio={metrics['glare_area_ratio']}")
    if flags['flag_mask_quality']:
        reasons.append('mask quality issue')
    reason = '; '.join(reasons) if reasons else 'No correction required'

    return {
        'raw_regression': None if v_reg is None else round(float(v_reg), 4),
        'raw_color_chart': None if v_color is None else round(float(v_color), 4),
        'fused': round(float(v_fused), 4),
        'corrected': round(float(v_corrected), 4),
        'snapped': None if snapped is None else round(float(snapped), 4),
        'alpha': round(float(alpha), 4),
        'correction_reason': reason,
        'boundary_distance': None if boundary_dist is None else round(float(boundary_dist), 4),
    }

# ───────────────────────────────
# Nearest-centroid utilities (Lab)
# ───────────────────────────────

def nearest_micro_centroid_lab(lab_obs):
    best_label, best_de = None, float('inf')
    for lbl, lab_c in MICROALBUMIN_CENTROIDS_LAB.items():
        de = _lab_distance(lab_obs, lab_c)
        if de < best_de: best_de, best_label = de, lbl
    return best_label, best_de

def nearest_creatinine_centroid_lab(lab_obs):
    best_label, best_de = None, float('inf')
    for lbl, lab_c in CREATININE_CENTROIDS_LAB.items():
        de = _lab_distance(lab_obs, lab_c)
        if de < best_de: best_de, best_label = de, lbl
    return best_label, best_de

def _creatinine_label_to_value(label):
    if label is None:
        return None
    if isinstance(label, (int, float)):
        return float(label)
    if isinstance(label, str):
        try:
            return float(label.split()[0])
        except (ValueError, IndexError):
            return None
    return None

def apply_low_end_snap(pod_type, rgb_obs, calibrated_value):
    lab_obs = _rgb_to_lab_triplet(rgb_obs)
    threshold = LOW_END_THRESHOLDS.get(pod_type)
    if threshold is None:
        return calibrated_value, None
    if calibrated_value is None or calibrated_value > threshold:
        return calibrated_value, None
    if pod_type == 'microalbumin':
        label, de = nearest_micro_centroid_lab(lab_obs)
        if label is not None and de <= MICRO_DELTAE_ACCEPT:
            return float(label), label
    elif pod_type == 'creatinine':
        label, de = nearest_creatinine_centroid_lab(lab_obs)
        if label is not None and de <= CREATININE_DELTAE_ACCEPT:
            snapped_value = _creatinine_label_to_value(label)
            if snapped_value is not None:
                return snapped_value, label
    return calibrated_value, None


def _safe_json_float(value, ndigits=4):
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _nearest_allowed_micro_legacy_bin(value):
    f = _safe_json_float(value)
    if f is None:
        return None
    return float(min(MICRO_LEGACY_RECOVERY_ALLOWED_BINS, key=lambda b: abs(float(b) - f)))


def _empty_legacy_candidate(version, reason):
    return {
        "version": version,
        "median_rgb": None,
        "median_lab": None,
        "continuous_albumin_value": None,
        "mapped_albumin_bin": None,
        "nearest_chart_bin": None,
        "nearest_chart_de": None,
        "low_end_snap_applied": False,
        "reason": reason,
    }


def estimate_microalbumin_v2_style(raw_np, wb_np, pod_mask_bool, regression_model=None):
    """
    Approximate V2 behavior:
    - Use gray-world white-balanced image path.
    - Erode microalbumin pod mask.
    - Extract median RGB from eroded WB pod pixels.
    - Convert RGB to RGB/HSV/Lab feature vector.
    - Run microalbumin regression model if available.
    - Apply legacy low-end snapping when estimate <= 30 mg/L and nearest low centroid is close.
    - Return both continuous estimate and snapped/bin estimate.
    """
    if wb_np is None:
        if raw_np is None:
            return _empty_legacy_candidate("v2_style", "missing_image")
        wb_np = gray_world_white_balance(np.asarray(raw_np, dtype=np.uint8))
    if pod_mask_bool is None:
        return _empty_legacy_candidate("v2_style", "missing_pod_mask")

    pod_eroded = eroded_mask(np.asarray(pod_mask_bool, dtype=bool))
    if int(pod_eroded.sum()) < MICRO_SHADE_MIN_PIXELS:
        return _empty_legacy_candidate("v2_style", "insufficient_mask_pixels")

    median_rgb_arr = masked_median_rgb(np.asarray(wb_np, dtype=np.uint8), pod_eroded)
    median_rgb = tuple(int(x) for x in median_rgb_arr.tolist())
    median_lab_arr = _rgb_to_lab_triplet(median_rgb)
    median_lab = tuple(_safe_json_float(x) for x in median_lab_arr.tolist())
    nearest_chart_bin, nearest_chart_de = nearest_micro_centroid_lab(median_lab_arr)

    continuous_value = None
    reason = "nearest_chart_fallback"
    if regression_model is not None:
        try:
            continuous_value = float(regression_model.predict([extract_features_from_rgb(median_rgb)])[0])
            reason = "regression_model"
        except Exception:
            continuous_value = None
            reason = "regression_failed_nearest_chart_fallback"

    if continuous_value is None:
        mapped_value = float(nearest_chart_bin) if nearest_chart_bin is not None else None
    else:
        snapped_value, low_end_label = apply_low_end_snap("microalbumin", median_rgb, continuous_value)
        if low_end_label is not None:
            mapped_value = float(snapped_value)
        else:
            mapped_value = _nearest_allowed_micro_legacy_bin(snapped_value)
        reason = f"{reason}_low_end_snap" if low_end_label is not None else reason

    mapped_bin = _nearest_allowed_micro_legacy_bin(mapped_value)
    return {
        "version": "v2_style",
        "median_rgb": median_rgb,
        "median_lab": median_lab,
        "continuous_albumin_value": _safe_json_float(continuous_value),
        "mapped_albumin_bin": mapped_bin,
        "nearest_chart_bin": None if nearest_chart_bin is None else int(nearest_chart_bin),
        "nearest_chart_de": _safe_json_float(nearest_chart_de),
        "low_end_snap_applied": bool("low_end_snap" in reason),
        "reason": reason,
    }


def estimate_microalbumin_v3_style(raw_np, wb_np, pod_mask_bool, regression_model=None):
    """
    Approximate V3 behavior using the same core extraction as V2, with a
    separate version label and trace fields for comparison.
    """
    candidate = dict(estimate_microalbumin_v2_style(raw_np, wb_np, pod_mask_bool, regression_model=regression_model))
    candidate["version"] = "v3_style"
    if candidate.get("reason"):
        candidate["reason"] = str(candidate["reason"]).replace("v2", "v3")
    return candidate


def _resolve_legacy_microalbumin_conflict(v2_bin, v3_bin):
    v2_bin = _nearest_allowed_micro_legacy_bin(v2_bin)
    v3_bin = _nearest_allowed_micro_legacy_bin(v3_bin)
    mean_candidate_bin = None
    if v2_bin is not None and v3_bin is not None:
        if float(v2_bin) == float(v3_bin):
            return v2_bin, v3_bin, float(v2_bin), "agreement", None
        mean_candidate_bin = (float(v2_bin) + float(v3_bin)) / 2.0
        return v2_bin, v3_bin, _nearest_allowed_micro_legacy_bin(mean_candidate_bin), "conflict_average_bin", mean_candidate_bin
    if v2_bin is not None or v3_bin is not None:
        return v2_bin, v3_bin, float(v2_bin if v2_bin is not None else v3_bin), "single_candidate_available", None
    return v2_bin, v3_bin, None, "no_candidate_available", None


def _select_higher_legacy_uacr(uacr_v2, uacr_v3, uacr_recovered):
    candidates = [
        ("average_bin_recovered", uacr_recovered, 0),
        ("v3_style", uacr_v3, 1),
        ("v2_style", uacr_v2, 2),
    ]
    numeric = []
    for source, value, tie_rank in candidates:
        f = _safe_json_float(value)
        if f is not None:
            numeric.append((f, -tie_rank, source))
    if not numeric:
        return None, None
    numeric.sort(reverse=True)
    selected_uacr, _, source = numeric[0]
    return selected_uacr, source


def stage_uacr_value(uacr_mg_g):
    if uacr_mg_g is None:
        return {"uacr_stage": "Unconfirmed", "uacr_stage_code": "unconfirmed"}
    if uacr_mg_g < 30:
        return {"uacr_stage": "A1 / normal to mildly increased", "uacr_stage_code": "A1"}
    if uacr_mg_g <= 300:
        return {"uacr_stage": "A2 / moderately increased", "uacr_stage_code": "A2"}
    return {"uacr_stage": "A3 / severely increased", "uacr_stage_code": "A3"}


def recover_microalbumin_from_legacy_when_unconfirmed(
    raw_np,
    wb_np,
    pod_mask_bool,
    creatinine_mg_dl,
    v4_guard,
    regression_model=None,
    allow_liberal=True,
):
    """
    Experimental fallback used only when V4 microalbumin result is unconfirmed.

    It computes V2/V3-style estimates, maps them to allowed bins, resolves
    conflicts by average-bin policy, computes direct UACR values, and selects
    the higher UACR for display. This function must not be used for normal V4
    exact/guarded/provisional/high-watch cases.
    """
    warning = (
        "Legacy recovery used liberal acceptance after V4 unconfirmed and selected "
        "the higher UACR among V2/V3/recovered candidates."
    )
    base = {
        "triggered_by": "v4_unconfirmed",
        "enabled": bool(ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY),
        "mode": MICRO_LEGACY_RECOVERY_MODE,
        "accepted": False,
        "v4_original_action": str((v4_guard or {}).get("action", "")),
        "v4_original_report_mode": str((v4_guard or {}).get("report_mode", "")),
        "v4_original_corrected_albumin_value": _safe_json_float((v4_guard or {}).get("corrected_albumin_value")),
        "v2_candidate": None,
        "v3_candidate": None,
        "v2_bin": None,
        "v3_bin": None,
        "mean_candidate_bin": None,
        "recovered_albumin_bin": None,
        "conflict_status": "not_evaluated",
        "uacr_v2": None,
        "uacr_v3": None,
        "uacr_recovered": None,
        "selected_uacr": None,
        "selected_uacr_source": None,
        "uacr_policy": "higher_uacr_selected",
        "final_report_mode": "unconfirmed",
        "final_action": "legacy_recovery_not_accepted",
        "warning": warning,
        "debug": {
            "accept_ood": bool(MICRO_LEGACY_RECOVERY_ACCEPT_OOD),
            "accept_low_confidence": bool(MICRO_LEGACY_RECOVERY_ACCEPT_LOW_CONFIDENCE),
            "min_chartlike_de": float(MICRO_LEGACY_RECOVERY_MIN_CHARTLIKE_DE),
            "conflict_policy": MICRO_LEGACY_RECOVERY_CONFLICT_POLICY,
            "allowed_bins": [int(x) for x in MICRO_LEGACY_RECOVERY_ALLOWED_BINS],
            "allow_liberal": bool(allow_liberal),
        },
    }
    if not ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY:
        base["final_action"] = "legacy_recovery_disabled"
        return base
    if str((v4_guard or {}).get("report_mode")) != "unconfirmed":
        base["triggered_by"] = "not_v4_unconfirmed"
        base["final_action"] = "legacy_recovery_skipped_non_unconfirmed_v4"
        return base
    if pod_mask_bool is None:
        base["final_action"] = "legacy_recovery_missing_mask"
        base["conflict_status"] = "no_candidate_available"
        return base
    try:
        valid_pixels = int(eroded_mask(np.asarray(pod_mask_bool, dtype=bool)).sum())
    except Exception:
        base["final_action"] = "legacy_recovery_invalid_mask"
        base["conflict_status"] = "no_candidate_available"
        return base
    if valid_pixels < MICRO_SHADE_MIN_PIXELS:
        base["final_action"] = "legacy_recovery_insufficient_mask_pixels"
        base["conflict_status"] = "no_candidate_available"
        return base
    creatinine = _safe_json_float(creatinine_mg_dl)
    if creatinine is None or creatinine <= 0:
        base["final_action"] = "legacy_recovery_invalid_creatinine"
        base["conflict_status"] = "no_candidate_available"
        return base

    try:
        v2_candidate = estimate_microalbumin_v2_style(raw_np, wb_np, pod_mask_bool, regression_model=regression_model)
        v3_candidate = estimate_microalbumin_v3_style(raw_np, wb_np, pod_mask_bool, regression_model=regression_model)
    except Exception as exc:
        base["v2_candidate"] = _empty_legacy_candidate("v2_style", f"fatal_exception: {type(exc).__name__}")
        base["v3_candidate"] = _empty_legacy_candidate("v3_style", f"fatal_exception: {type(exc).__name__}")
        base["final_action"] = "legacy_recovery_feature_extraction_exception"
        base["conflict_status"] = "no_candidate_available"
        return base

    v2_bin, v3_bin, recovered_bin, conflict_status, mean_candidate_bin = _resolve_legacy_microalbumin_conflict(
        (v2_candidate or {}).get("mapped_albumin_bin"),
        (v3_candidate or {}).get("mapped_albumin_bin"),
    )
    uacr_v2 = None if v2_bin is None else 100.0 * float(v2_bin) / creatinine
    uacr_v3 = None if v3_bin is None else 100.0 * float(v3_bin) / creatinine
    uacr_recovered = None if recovered_bin is None else 100.0 * float(recovered_bin) / creatinine
    selected_uacr, selected_source = _select_higher_legacy_uacr(uacr_v2, uacr_v3, uacr_recovered)
    accepted = bool(allow_liberal and recovered_bin is not None and selected_uacr is not None)

    base.update({
        "accepted": accepted,
        "v2_candidate": v2_candidate,
        "v3_candidate": v3_candidate,
        "v2_bin": _safe_json_float(v2_bin),
        "v3_bin": _safe_json_float(v3_bin),
        "mean_candidate_bin": _safe_json_float(mean_candidate_bin),
        "recovered_albumin_bin": _safe_json_float(recovered_bin),
        "conflict_status": conflict_status,
        "uacr_v2": _safe_json_float(uacr_v2),
        "uacr_v3": _safe_json_float(uacr_v3),
        "uacr_recovered": _safe_json_float(uacr_recovered),
        "selected_uacr": _safe_json_float(selected_uacr),
        "selected_uacr_source": selected_source,
        "final_report_mode": "legacy_recovered" if accepted else "unconfirmed",
        "final_action": "legacy_recovery_after_v4_unconfirmed" if accepted else "legacy_recovery_no_candidate_available",
    })
    return base

# ───────────────────────────────
# Color Charts & Reference Values
# ───────────────────────────────
POD_COLOR_CHART = {
    'pod2': {label: tuple(map(int, MICROALBUMIN_CENTROIDS[label])) for label in MICROALBUMIN_CENTROIDS},
    'pod1': {label: tuple(map(int, CREATININE_CENTROIDS[label])) for label in CREATININE_CENTROIDS}
}

REFERENCE_VALUES = {
    'pod2': {**{str(label): label for label in MICROALBUMIN_CENTROIDS}, '1800': 1800},
    'pod1': {'10 (0.1)': 10, '25 (0.25)': 25, '50 (0.5)': 50,
             '100 (1.0)': 100, '150 (1.5)': 150,
             '200 (2.0)': 200, '300 (4.0)': 300}
}

# ───────────────────────────────
# Feature Extraction & Calibration
# ───────────────────────────────

def extract_features_from_rgb(rgb_triplet):
    rgb_arr = np.uint8([[rgb_triplet]])
    lab = rgb2lab(rgb_arr).reshape(3,)
    hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV).reshape(3,)
    return list(rgb_triplet) + list(hsv) + list(lab)

def get_calibrated_value(rgb_triplet, pod_type):
    if model_creat is None or model_micro is None:
        logging.warning("get_calibrated_value called but models are not loaded.")
        return None
    features = extract_features_from_rgb(rgb_triplet)
    try:
        if pod_type == 'creatinine':
            return float(model_creat.predict([features])[0])
        if pod_type == 'microalbumin':
            return float(model_micro.predict([features])[0])
    except Exception as e:
        logging.error(f"Error during model.predict() for {pod_type}.", exc_info=True)
        return None

def find_closest_reference_value_label(val, pod_type):
    if val is None:
        if pod_type == 'pod2':
            return 'Unconfirmed'
        return '300 (4.0)'
    chart, best, md = REFERENCE_VALUES.get(pod_type, {}), None, float('inf')
    for lbl, rv in chart.items():
        diff = abs(val - rv)
        if diff < md: md, best = diff, lbl
    return best or f"{val:.2f}"

# ───────────────────────────────
# Composite Visualization
# ───────────────────────────────

def save_composite_visual(raw_img, pod1_region, pod2_region,
                          p1_mean_display, p2_mean_display,
                          calibrated_p1, calibrated_p2,
                          uacr_display,
                          save_path,
                          pod_quality=None,
                          uacr_confidence=None,
                          uacr_report_payload=None):
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    axs[0].imshow(raw_img, interpolation='nearest'); axs[0].axis('off'); axs[0].set_title('Original')

    patch1 = np.ones((50, 50, 3), np.uint8) * p1_mean_display.reshape(1, 1, 3)
    disp1 = find_closest_reference_value_label(calibrated_p1, 'pod1')
    p1_quality_suffix = ""
    if pod_quality and pod_quality.get('creatinine'):
        q = pod_quality['creatinine']
        p1_quality_suffix = f"\nConf {q.get('confidence_pct', 'NA')}% ({q.get('confidence_bucket', 'NA')})"
    creatinine_si_display = ""
    if pod_quality and pod_quality.get('creatinine'):
        cr_si = (pod_quality['creatinine'].get('creatinine_si') or {})
        if cr_si.get('creatinine_si_display'):
            creatinine_si_display = f"\n{cr_si.get('creatinine_si_display')}"
    axs[1].imshow(patch1, interpolation='nearest'); axs[1].axis('off'); axs[1].set_title(f"Creatinine\n{disp1} mg/dL{creatinine_si_display}\nRGB{tuple(p1_mean_display)}{p1_quality_suffix}")

    patch2 = np.ones((50, 50, 3), np.uint8) * p2_mean_display.reshape(1, 1, 3)
    disp2 = find_closest_reference_value_label(calibrated_p2, 'pod2')
    p2_quality_suffix = ""
    p2_guard_suffix = ""
    if pod_quality and pod_quality.get('microalbumin'):
        q = pod_quality['microalbumin']
        p2_quality_suffix = f"\nConf {q.get('confidence_pct', 'NA')}% ({q.get('confidence_bucket', 'NA')})"
        shade_guard = q.get('shade_sanity_check')
        if isinstance(shade_guard, dict):
            action = shade_guard.get('action', 'unknown')
            low_pct = 100.0 * float(shade_guard.get('low_pixel_fraction', 0.0))
            aqua_pct = 100.0 * float(shade_guard.get('aqua_pixel_fraction', 0.0))
            low_visual = shade_guard.get('low_candidate_visual_name', 'n/a')
            low_confirmed = bool(shade_guard.get('low_shade_confirmed', False))
            weak_aqua = 'yes' if shade_guard.get('weak_aqua_low_compatible') else 'no'
            moderate_aqua = 'yes' if shade_guard.get('moderate_aqua_present') else 'no'
            strong_aqua = 'yes' if shade_guard.get('strong_aqua_confirmed') else 'no'
            overbright = 'yes' if shade_guard.get('overbright_not_chart_like') else 'no'
            median_l = float(shade_guard.get('median_L', 0.0))
            low_visual_line = (
                f"Confirmed low visual: {low_visual}"
                if low_confirmed else
                f"Nearest low comparator: {low_visual}"
            )
            report_mode = q.get("microalbumin_report_mode", "exact")
            if report_mode == "provisional_range":
                disp2 = q.get("microalbumin_display_text", "Provisional / retest")
            elif report_mode == "legacy_recovered":
                recovered = q.get("final_display_value_after_legacy_recovery", q.get("final_display_value"))
                disp2 = q.get("microalbumin_display_text", f"Legacy recovered: {recovered} mg/L")
            elif report_mode == "unconfirmed":
                disp2 = "Unconfirmed / retake image"
            else:
                disp2 = q.get("microalbumin_display_text", disp2)
            micro_si = q.get("microalbumin_si") or {}
            if micro_si.get("microalbumin_si_display") and report_mode != "unconfirmed":
                disp2 = f"{disp2}\n{micro_si.get('microalbumin_si_display')}"
            guard_scenario = (q.get("guarded_uacr_scenario") or {}).get("provisional_guard_scenario", "n/a")
            recovery_line = ""
            if report_mode == "legacy_recovered":
                result = q.get("legacy_recovery_result") or {}
                recovery_line = (
                    f"\nV4: Unconfirmed"
                    f"\nRecovery: V2/V3 liberal average-bin"
                    f"\nUACR source: {result.get('selected_uacr_source', 'n/a')}"
                )
            p2_guard_suffix = (
                f"\nShade guard: {action}"
                f"\nLow evidence: {low_pct:.1f}%"
                f"\nAqua evidence: {aqua_pct:.1f}%"
                f"\nWeak aqua low-compatible: {weak_aqua}"
                f"\nModerate aqua: {moderate_aqua}"
                f"\nStrong aqua: {strong_aqua}"
                f"\nOverbright: {overbright}"
                f"\nMedian L*: {median_l:.1f}"
                f"\n{low_visual_line}"
                f"\nGuard scenario: {guard_scenario}"
                f"{recovery_line}"
            )
    axs[2].imshow(patch2, interpolation='nearest'); axs[2].axis('off'); axs[2].set_title(f"Microalbumin\n{disp2}\nRGB{tuple(p2_mean_display)}{p2_quality_suffix}{p2_guard_suffix}")

    if uacr_display:
        color = 'darkgreen' if 'A1' in uacr_display else '#D98E04' if 'A2' in uacr_display else 'darkred'
        conf_text = f"\nEvidence confidence: {uacr_confidence}%" if uacr_confidence is not None else ""
        uacr_mode = (uacr_report_payload or {}).get("uacr_report_mode", "exact")
        if uacr_mode == "provisional_range":
            guarded_range = (uacr_report_payload or {}).get("uacr_reference_range", "Unavailable")
            albumin_range = (uacr_report_payload or {}).get("uacr_guarded_albumin_range_mg_l")
            albumin_range_txt = "Unavailable"
            if albumin_range:
                albumin_range_txt = f"{albumin_range[0]:.0f}\u2013{albumin_range[1]:.0f} mg/L"
            acr_si_range_display = ((uacr_report_payload or {}).get("acr_si_range") or {}).get("acr_si_range_display")
            acr_si_line = f"\nACR SI range: {acr_si_range_display}" if acr_si_range_display else ""
            suptitle = f"UACR Result\n{uacr_display}\nGuarded UACR range: {guarded_range}{acr_si_line}\nAlbumin guard range: {albumin_range_txt}\nExact albumin: not finalized{conf_text}"
        elif uacr_mode == "unconfirmed":
            suptitle = f"UACR Result\nUnconfirmed / retake image{conf_text}"
        elif uacr_mode == "legacy_recovered":
            val = (uacr_report_payload or {}).get("uacr_value")
            stage_code = (uacr_report_payload or {}).get("uacr_stage_code")
            source = (uacr_report_payload or {}).get("legacy_recovered_uacr_source")
            source_note = "\nUACR selected from higher legacy candidate" if source in ("v2_style", "v3_style") else ""
            acr_si_display = ((uacr_report_payload or {}).get("acr_si") or {}).get("acr_si_display")
            val_line = f"{val:.2f} mg/g" if val is not None else "Unavailable"
            si_line = f" / {acr_si_display}" if acr_si_display else ""
            suptitle = f"UACR Result\nLegacy recovered: {val_line}{si_line}, {stage_code or 'Unconfirmed'}{source_note}\nNot V4 confirmed{conf_text}"
        else:
            ref_range = (uacr_report_payload or {}).get("uacr_reference_range")
            val = (uacr_report_payload or {}).get("uacr_value")
            acr_si_display = ((uacr_report_payload or {}).get("acr_si") or {}).get("acr_si_display")
            si_line = f"\nACR SI: {acr_si_display}" if acr_si_display else ""
            val_line = f"\nValue: {val:.2f} mg/G" if val is not None else ""
            ref_line = f"\nReference Range: {ref_range}" if ref_range else ""
            suptitle = f"UACR Result\n{uacr_display}{val_line}{si_line}{ref_line}{conf_text}"
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', color=color, y=0.97)

    plt.tight_layout(rect=[0, 0.03, 1, 0.82])
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    finally:
        plt.close(fig)

# ───────────────────────────────
# UACR Calculation & Categorization
# ───────────────────────────────

def calculate_uacr_and_category(albumin_mg_l, creatinine_mg_dl):
    if albumin_mg_l is None or creatinine_mg_dl is None or creatinine_mg_dl <= 0:
        return None, "Unconfirmed", "Retest / unavailable", "Unconfirmed"

    uacr = 100.0 * albumin_mg_l / creatinine_mg_dl
    uacr_rounded = round(uacr, 2)

    if uacr < 30:
        stage = "A1 Proteinuria"
        reference_range = "< 30 mg/G"
    elif 30 <= uacr <= 300:
        stage = "A2 Proteinuria"
        reference_range = "30 - 300 mg/G"
    else:
        stage = "A3 Proteinuria"
        reference_range = "> 300 mg/G"

    display_text = (
        f"{stage}\n"
        f"Value: {uacr_rounded:.2f} mg/G\n"
        f"Reference Range: {reference_range}"
    )

    return uacr_rounded, stage, reference_range, display_text


def build_uacr_trace(albumin_used_mg_l, creatinine_used_mg_dl, uacr_value, source_tag):
    return {
        'albumin_used_mg_l': None if albumin_used_mg_l is None else round(float(albumin_used_mg_l), 4),
        'creatinine_used_mg_dl': None if creatinine_used_mg_dl is None else round(float(creatinine_used_mg_dl), 4),
        'uacr_formula_mg_g': uacr_value,
        'source': source_tag,
    }

# ───────────────────────────────
# Main Inference
# ───────────────────────────────

def process_image_and_get_pods(image_path, model, device):
    img_pil = Image.open(image_path).convert('RGB')
    if img_pil.width > img_pil.height:
        img_pil = img_pil.rotate(90, expand=True)
    raw_np = np.array(img_pil)

    model_input = val_tf(image=raw_np)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(model_input)
        preds  = logits.argmax(1).cpu().numpy()[0]
    mask = cv2.resize(preds.astype(np.uint8), raw_np.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    def mean_rgb_raw(img_uint8, m_bool):
        if not m_bool.any(): return np.zeros(3, dtype=np.uint8)
        return np.round(img_uint8[m_bool].mean(0)).astype(np.uint8)

    p1_mean_raw = mean_rgb_raw(raw_np, (mask == POD1_IDX))
    p2_mean_raw = mean_rgb_raw(raw_np, (mask == POD2_IDX))

    wb_np = gray_world_white_balance(raw_np)
    p1_mean_ui = masked_median_rgb(wb_np, eroded_mask((mask == POD1_IDX)))
    p2_mean_ui = masked_median_rgb(wb_np, eroded_mask((mask == POD2_IDX)))

    pod1_mask = (mask == POD1_IDX)
    pod2_mask = (mask == POD2_IDX)
    pod1_eroded = eroded_mask(pod1_mask)
    pod2_eroded = eroded_mask(pod2_mask)

    pod1_mask_quality = _compute_mask_quality(pod1_mask, raw_np.shape[:2], QUALITY_THRESHOLDS)
    pod2_mask_quality = _compute_mask_quality(pod2_mask, raw_np.shape[:2], QUALITY_THRESHOLDS)
    pod1_metrics = _compute_quality_metrics(wb_np, pod1_eroded)
    pod2_metrics = _compute_quality_metrics(wb_np, pod2_eroded)
    pod1_flags = _derive_quality_flags(pod1_metrics, pod1_mask_quality, QUALITY_THRESHOLDS)
    pod2_flags = _derive_quality_flags(pod2_metrics, pod2_mask_quality, QUALITY_THRESHOLDS)
    pod1_conf = _confidence_from_quality(pod1_metrics, pod1_flags)
    pod2_conf = _confidence_from_quality(pod2_metrics, pod2_flags)

    c1 = get_calibrated_value(tuple(p1_mean_ui), 'creatinine')
    c2 = get_calibrated_value(tuple(p2_mean_ui), 'microalbumin')
    c1_color, c1_color_label = _estimate_from_chart('creatinine', tuple(p1_mean_ui))
    c2_color, c2_color_label = _estimate_from_chart('microalbumin', tuple(p2_mean_ui))

    c1_quality_trace = _apply_quality_corrections('creatinine', c1, c1_color, pod1_conf, pod1_flags, pod1_metrics)
    c2_quality_trace = _apply_quality_corrections('microalbumin', c2, c2_color, pod2_conf, pod2_flags, pod2_metrics)

    c1_pre_snap = c1_quality_trace['snapped'] if c1_quality_trace['snapped'] is not None else c1
    c2_pre_snap = c2_quality_trace['snapped'] if c2_quality_trace['snapped'] is not None else c2

    c1_snapped, c1_low_end_label = apply_low_end_snap('creatinine', tuple(p1_mean_ui), c1_pre_snap)
    c2_snapped, c2_low_end_label = apply_low_end_snap('microalbumin', tuple(p2_mean_ui), c2_pre_snap)
    albumin_shade_guard = microalbumin_shade_sanity_check(
        wb_np,
        pod2_mask,
        c2_snapped,
        current_albumin_label=c2_low_end_label,
        current_confidence=pod2_conf.get('confidence'),
        allow_unconfirmed=True,
    )
    albumin_guarded_scenario = derive_microalbumin_guarded_uacr_scenario(
        shade_guard=albumin_shade_guard,
        creatinine_mg_dl=c1_snapped,
        original_albumin_value=c2_snapped,
        exact_albumin_value=albumin_shade_guard.get("corrected_albumin_value"),
    )
    microalbumin_report_display = choose_microalbumin_report_display(
        exact_value=albumin_shade_guard.get("corrected_albumin_value"),
        shade_guard=albumin_shade_guard,
        guarded_scenario=albumin_guarded_scenario,
    )
    c2_final_exact = albumin_shade_guard.get("corrected_albumin_value")
    c2_report_mode = microalbumin_report_display["microalbumin_report_mode"]
    c2_display_text = microalbumin_report_display["microalbumin_display_text"]
    c2_range = microalbumin_report_display["microalbumin_range_mg_l"]

    microalbumin_report_mode_before_legacy_recovery = c2_report_mode
    final_display_value_before_legacy_recovery = c2_final_exact
    legacy_recovery_attempted = bool(
        ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY
        and albumin_shade_guard.get("report_mode") == "unconfirmed"
    )
    legacy_recovery_result = None
    if legacy_recovery_attempted:
        legacy_recovery_result = recover_microalbumin_from_legacy_when_unconfirmed(
            raw_np,
            wb_np,
            pod2_mask,
            c1_snapped,
            albumin_shade_guard,
            regression_model=model_micro,
            allow_liberal=True,
        )
        if legacy_recovery_result.get("accepted"):
            c2_final_exact = legacy_recovery_result.get("recovered_albumin_bin")
            c2_report_mode = "legacy_recovered"
            c2_display_text = (
                f"Legacy recovered: {float(c2_final_exact):.0f} mg/L\n"
                "V4: Unconfirmed\n"
                "Recovery: V2/V3 liberal average-bin"
            )
            c2_range = None

    final_display_value_after_legacy_recovery = c2_final_exact
    microalbumin_report_mode_after_legacy_recovery = c2_report_mode

    # Preserve legacy behavior trace (continuous calibrated values) and corrected
    # behavior trace (snapped/displayed values) to avoid disruption in existing flow.
    uacr_legacy_value, _, _, _ = calculate_uacr_and_category(c2, c1)
    guarded_uacr_range_mg_g = None
    guarded_albumin_range_mg_l = None
    if microalbumin_report_mode_before_legacy_recovery in ("exact", "guarded_exact"):
        uacr_report_mode_before_legacy_recovery = "exact"
    elif microalbumin_report_mode_before_legacy_recovery == "high_watch":
        uacr_report_mode_before_legacy_recovery = "high_watch"
    elif microalbumin_report_mode_before_legacy_recovery == "provisional_range":
        uacr_report_mode_before_legacy_recovery = "provisional_range"
    else:
        uacr_report_mode_before_legacy_recovery = "unconfirmed"
    if c2_report_mode in ("exact", "guarded_exact") and c2_final_exact is not None:
        uacr_value, uacr_stage, uacr_range, uacr_display = calculate_uacr_and_category(c2_final_exact, c1_snapped)
        c2_report_mode_for_uacr = "exact"
    elif c2_report_mode == "high_watch" and c2_final_exact is not None:
        uacr_value, uacr_stage, uacr_range, exact_uacr_display = calculate_uacr_and_category(c2_final_exact, c1_snapped)
        uacr_display = f"{exact_uacr_display} (High-watch / retest recommended)"
        c2_report_mode_for_uacr = "high_watch"
    elif c2_report_mode == "legacy_recovered":
        uacr_value = legacy_recovery_result.get("selected_uacr") if legacy_recovery_result else None
        staged = stage_uacr_value(uacr_value)
        uacr_stage = staged["uacr_stage"]
        uacr_stage_code = staged["uacr_stage_code"]
        uacr_range = uacr_stage
        uacr_display = f"{float(uacr_value):.2f} mg/g, {uacr_stage_code}" if uacr_value is not None else "Unconfirmed / retake image"
        source = legacy_recovery_result.get("selected_uacr_source") if legacy_recovery_result else None
        if source in ("v2_style", "v3_style"):
            uacr_display = f"{uacr_display} (UACR selected from higher legacy candidate)"
        guarded_uacr_range_mg_g = None
        guarded_albumin_range_mg_l = None
        c2_report_mode_for_uacr = "legacy_recovered"
    elif c2_report_mode == "provisional_range":
        uacr_value = None
        uacr_stage = albumin_guarded_scenario["provisional_uacr_stage"]
        uacr_range = albumin_guarded_scenario["provisional_uacr_display"]
        uacr_display = albumin_guarded_scenario["provisional_uacr_display"]
        guarded_uacr_range_mg_g = albumin_guarded_scenario["provisional_uacr_range_mg_g"]
        guarded_albumin_range_mg_l = albumin_guarded_scenario["provisional_albumin_range_mg_l"]
        c2_report_mode_for_uacr = "provisional_range"
    else:
        uacr_value = None
        uacr_stage = "Unconfirmed / retake image"
        uacr_range = "Unavailable"
        uacr_display = "Unconfirmed / retake image"
        c2_report_mode_for_uacr = "unconfirmed"
    unit_systems = {
        "conventional": {
            "creatinine_unit": "mg/dL",
            "microalbumin_unit": "mg/L",
            "uacr_unit": "mg/g",
        },
        "si": {
            "creatinine_unit": "µmol/L",
            "creatinine_ratio_unit": "mmol/L",
            "microalbumin_unit": "g/L",
            "microalbumin_equivalent_unit": "µg/mL",
            "acr_unit": "mg/mmol",
        },
    }
    creatinine_si = convert_creatinine_to_si(c1_snapped)
    microalbumin_si = convert_microalbumin_to_si(c2_final_exact) if c2_final_exact is not None else None
    acr_si = None
    acr_si_range = None
    if c2_report_mode_for_uacr in ("exact", "guarded_exact", "high_watch", "legacy_recovered") and c2_final_exact is not None:
        acr_si = calculate_acr_si(c2_final_exact, c1_snapped)
    elif c2_report_mode_for_uacr == "provisional_range":
        acr_si = _unconfirmed_acr_si()
        acr_si_range = calculate_acr_si_range(guarded_albumin_range_mg_l, c1_snapped)
        if guarded_albumin_range_mg_l is not None:
            low_albumin, high_albumin = guarded_albumin_range_mg_l
            microalbumin_si = {
                "microalbumin_range_mg_l": (round(float(low_albumin), 2), round(float(high_albumin), 2)),
                "microalbumin_range_g_l": (round(float(low_albumin) / 1000.0, 3), round(float(high_albumin) / 1000.0, 3)),
                "microalbumin_si_display": f"{float(low_albumin) / 1000.0:.3f}–{float(high_albumin) / 1000.0:.3f} g/L",
                "microalbumin_equivalent_display": f"{float(low_albumin):.2f}–{float(high_albumin):.2f} µg/mL",
            }
    else:
        acr_si = _unconfirmed_acr_si()
    uacr_confidence = round(min(pod1_conf['confidence'], pod2_conf['confidence']) * 100.0, 1)
    uacr_conf_bucket = 'High' if uacr_confidence >= 85 else 'Moderate' if uacr_confidence >= 65 else 'Low'
    uacr_delta = None if (uacr_legacy_value is None or uacr_value is None) else round(uacr_legacy_value - uacr_value, 2)
    legacy_stage = calculate_uacr_and_category(c2, c1)[1]
    stage_adjustment_reason = None
    if legacy_stage != uacr_stage:
        stage_adjustment_reason = "Stage adjusted due to low evidence confidence"

    if uacr_delta is not None and abs(uacr_delta) > 1.0:
        logging.warning(
            "UACR consistency warning: legacy/calibrated=%.2f, corrected/snapped=%.2f, delta=%.2f",
            uacr_legacy_value,
            uacr_value,
            uacr_delta,
        )

    out_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(os.path.basename(image_path))[0]}_{uuid.uuid4().hex[:6]}.png"
    fpath = os.path.join(out_dir, fname)
    
    save_composite_visual(
        raw_np,
        None,
        None,
        p1_mean_ui,
        p2_mean_ui,
        c1_snapped,
        c2_final_exact,
        uacr_display,
        fpath,
        pod_quality={
            'creatinine': {**pod1_conf, 'creatinine_si': creatinine_si},
            'microalbumin': {
                **pod2_conf,
                'final_display_value_before_shade_guard': c2_snapped,
                'final_display_value': c2_final_exact,
                'shade_sanity_check': albumin_shade_guard,
                'guarded_uacr_scenario': albumin_guarded_scenario,
                'microalbumin_report_mode': c2_report_mode,
                'microalbumin_provisional_range_mg_l': albumin_shade_guard.get('provisional_albumin_range_mg_l'),
                'microalbumin_display_text': c2_display_text,
                'microalbumin_si': microalbumin_si,
                'legacy_recovery_attempted': legacy_recovery_attempted,
                'legacy_recovery_enabled': ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY,
                'legacy_recovery_mode': MICRO_LEGACY_RECOVERY_MODE,
                'legacy_recovery_result': legacy_recovery_result,
                'final_display_value_before_legacy_recovery': final_display_value_before_legacy_recovery,
                'final_display_value_after_legacy_recovery': final_display_value_after_legacy_recovery,
                'microalbumin_report_mode_before_legacy_recovery': microalbumin_report_mode_before_legacy_recovery,
                'microalbumin_report_mode_after_legacy_recovery': microalbumin_report_mode_after_legacy_recovery,
                'microalbumin_reporting_note': 'V4 returned unconfirmed. Experimental V2/V3 legacy recovery was attempted. Result is liberal and must be interpreted as legacy-recovered, not V4-confirmed.' if legacy_recovery_attempted else 'Microalbumin value was finalized, guarded, provisional, high-watch, or unconfirmed according to shade evidence.',
            },
        },
        uacr_confidence=uacr_confidence,
        uacr_report_payload={
            "uacr_report_mode": c2_report_mode_for_uacr,
            "uacr_value": uacr_value,
            "uacr_reference_range": uacr_range,
            "uacr_guarded_albumin_range_mg_l": guarded_albumin_range_mg_l,
            "uacr_guarded_range_mg_g": guarded_uacr_range_mg_g,
            "uacr_warning": "Microalbumin is high-watch; retest recommended." if c2_report_mode_for_uacr == "high_watch" else None,
            "uacr_stage_code": locals().get("uacr_stage_code"),
            "legacy_recovered_uacr_source": (legacy_recovery_result or {}).get("selected_uacr_source") if legacy_recovery_result else None,
            "acr_si": acr_si,
            "acr_si_range": acr_si_range,
        },
    )
    
    return {
        'composite_img': fname,
        'unit_systems': unit_systems,
        'creatinine_si': creatinine_si,
        'microalbumin_si': microalbumin_si,
        'acr_si': acr_si,
        'acr_si_range': acr_si_range,
        # Corrected value aligned to displayed snapped analyte values.
        'uacr_report_mode': c2_report_mode_for_uacr,
        'uacr_value': uacr_value,
        'uacr_category': uacr_display,
        'uacr_stage': uacr_stage,
        'uacr_reference_range': uacr_range,
        'uacr_display': uacr_display,
        'uacr_guarded_range_mg_g': guarded_uacr_range_mg_g,
        'uacr_guarded_albumin_range_mg_l': guarded_albumin_range_mg_l,
        'uacr_retest_recommended': bool(microalbumin_report_display["retest_recommended"]),
        'uacr_warning': 'Microalbumin is high-watch; retest recommended.' if c2_report_mode_for_uacr == 'high_watch' else None,
        'uacr_reporting_note': 'Provisional UACR ranges are triage-only and require retest when exact microalbumin is not finalized.',
        'uacr_report_mode_before_legacy_recovery': uacr_report_mode_before_legacy_recovery,
        'uacr_report_mode_after_legacy_recovery': c2_report_mode_for_uacr,
        'legacy_recovered_uacr_value': (legacy_recovery_result or {}).get('selected_uacr') if legacy_recovery_result else None,
        'legacy_recovered_uacr_source': (legacy_recovery_result or {}).get('selected_uacr_source') if legacy_recovery_result else None,
        'legacy_recovery_warning': 'Legacy recovery used liberal acceptance after V4 unconfirmed and selected the higher UACR among V2/V3/recovered candidates.' if legacy_recovery_attempted else None,
        # Traceability fields to preserve legacy vs corrected outputs.
        'uacr_legacy_value': uacr_legacy_value,
        'uacr_corrected_value': uacr_value,
        'uacr_delta_legacy_minus_corrected': uacr_delta,
        'uacr_legacy_trace': build_uacr_trace(c2, c1, uacr_legacy_value, 'legacy_calibrated_continuous'),
        'uacr_corrected_trace': (
            build_uacr_trace(c2_final_exact, c1_snapped, uacr_value, 'corrected_exact_with_microalbumin_shade_guard')
            if c2_report_mode_for_uacr in ("exact", "high_watch") else
            {
                "trace_type": "legacy_recovered_after_v4_unconfirmed",
                "albumin_value": c2_final_exact,
                "creatinine_mg_dl": c1_snapped,
                "uacr_value": uacr_value,
                "selected_uacr_source": (legacy_recovery_result or {}).get("selected_uacr_source"),
                "source_guard_action": albumin_shade_guard.get("action"),
            } if c2_report_mode_for_uacr == "legacy_recovered" else
            {
                "trace_type": "provisional_range_with_microalbumin_shade_guard",
                "albumin_range_mg_l": guarded_albumin_range_mg_l,
                "creatinine_mg_dl": c1_snapped,
                "uacr_range_mg_g": guarded_uacr_range_mg_g,
                "provisional_stage": uacr_stage,
                "reason": albumin_guarded_scenario.get("provisional_reason"),
                "source_guard_action": albumin_shade_guard.get("action"),
            } if c2_report_mode_for_uacr == "provisional_range" else {
                "trace_type": "unconfirmed_with_microalbumin_shade_guard",
                "albumin_value": None,
                "creatinine_mg_dl": c1_snapped,
                "uacr_value": None,
                "stage": "Unconfirmed / retake image",
                "reason": albumin_shade_guard.get("guard_reason"),
                "source_guard_action": albumin_shade_guard.get("action"),
            }
        ),
        # Quality and confidence payload for on-screen evidence display.
        'pod_quality': {
            'creatinine': {
                **pod1_conf,
                **pod1_flags,
                **pod1_metrics,
                **pod1_mask_quality,
                'trace': c1_quality_trace,
                'raw_regression_value': c1,
                'raw_color_chart_value': c1_color,
                'raw_color_chart_label': c1_color_label,
                'final_display_value': c1_snapped,
                'creatinine_si': creatinine_si,
                'low_end_snap_label': c1_low_end_label,
            },
            'microalbumin': {
                **pod2_conf,
                **pod2_flags,
                **pod2_metrics,
                **pod2_mask_quality,
                'trace': c2_quality_trace,
                'raw_regression_value': c2,
                'raw_color_chart_value': c2_color,
                'raw_color_chart_label': c2_color_label,
                'final_display_value_before_shade_guard': c2_snapped,
                'final_display_value': c2_final_exact,
                'microalbumin_si': microalbumin_si,
                'low_end_snap_label': c2_low_end_label,
                'shade_sanity_check': albumin_shade_guard,
                'guarded_uacr_scenario': albumin_guarded_scenario,
                'microalbumin_report_mode': c2_report_mode,
                'microalbumin_provisional_range_mg_l': albumin_shade_guard.get('provisional_albumin_range_mg_l'),
                'microalbumin_reporting_note': 'V4 returned unconfirmed. Experimental V2/V3 legacy recovery was attempted. Result is liberal and must be interpreted as legacy-recovered, not V4-confirmed.' if legacy_recovery_attempted else 'Microalbumin value was finalized, guarded, provisional, high-watch, or unconfirmed according to shade evidence.',
                'microalbumin_display_text': c2_display_text,
                'legacy_recovery_attempted': legacy_recovery_attempted,
                'legacy_recovery_enabled': ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY,
                'legacy_recovery_mode': MICRO_LEGACY_RECOVERY_MODE,
                'legacy_recovery_result': legacy_recovery_result,
                'final_display_value_before_legacy_recovery': final_display_value_before_legacy_recovery,
                'final_display_value_after_legacy_recovery': final_display_value_after_legacy_recovery,
                'microalbumin_report_mode_before_legacy_recovery': microalbumin_report_mode_before_legacy_recovery,
                'microalbumin_report_mode_after_legacy_recovery': microalbumin_report_mode_after_legacy_recovery,
                'microalbumin_report_range_mg_l': c2_range,
                'retest_recommended': microalbumin_report_display["retest_recommended"],
                'reporting_note': 'Exact microalbumin value is not finalized when guard evidence is unconfirmed or ambiguous; a provisional range is shown for A1/A2 triage only.',
            },
        },
        'uacr_confidence_pct': uacr_confidence,
        'uacr_confidence_bucket': uacr_conf_bucket,
        'uacr_stage_adjustment_reason': stage_adjustment_reason,
        'uacr_trace_quality': {
            'legacy_stage': legacy_stage,
            'corrected_stage': uacr_stage,
            'raw_formula_value': uacr_legacy_value,
            'corrected_formula_value': uacr_value,
            'uacr_confidence_pct': uacr_confidence,
        },
        'pod_color_trace': {
            'creatinine': {
                'raw_mean_rgb': tuple(map(int, p1_mean_raw.tolist())),
                'display_mean_rgb': tuple(map(int, p1_mean_ui.tolist())),
            },
            'microalbumin': {
                'raw_mean_rgb': tuple(map(int, p2_mean_raw.tolist())),
                'display_mean_rgb': tuple(map(int, p2_mean_ui.tolist())),
            },
        },
    }
