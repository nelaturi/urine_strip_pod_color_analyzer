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
MICRO_SHADE_PIXEL_DE_ACCEPT = 9.0
MICRO_SHADE_MEDIAN_LOW_DE_ACCEPT = 8.0
MICRO_SHADE_ADVANTAGE_MARGIN = 1.5
MICRO_LOW_SHADE_FRACTION_MIN = 0.50
MICRO_AQUA_SHADE_FRACTION_MIN = 0.05
MICRO_AQUA_SHADE_STRONG_FRACTION = 0.12
MICRO_SHADE_MIN_PIXELS = 50

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


def microalbumin_shade_sanity_check(
    img_rgb_uint8,
    pod_mask_bool,
    current_albumin_value,
    current_albumin_label=None,
):
    """
    Conservative post-hoc visual sanity check for the microalbumin pod.

    This does not replace regression or centroid classification. It exists to prevent
    false upward microalbumin labelling when the pod is visually in the 10/30 mg/L
    low-shade region and lacks 80/150 mg/L aqua-blue evidence.
    """
    guard_applied = False
    corrected_albumin_value = current_albumin_value
    action = "unchanged_not_evaluated"
    guard_reason = ""

    default_low_candidate_class = 10
    default_aqua_candidate_class = 80

    def _build_response(
        action_value,
        guard_reason_value,
        original_value,
        corrected_value,
        low_candidate_class_mg_l=10,
        aqua_candidate_class_mg_l=80,
        median_de_10=0.0,
        median_de_30=0.0,
        median_de_80=0.0,
        median_de_150=0.0,
        median_low_de=0.0,
        median_aqua_de=0.0,
        low_margin=0.0,
        aqua_margin=0.0,
        low_pixel_fraction=0.0,
        aqua_pixel_fraction=0.0,
        high_value_confirmed_by_aqua=False,
        confidence_bucket="Low",
        guard_applied_value=False,
    ):
        low_ref = MICROALBUMIN_LOW_SHADE_REFS[int(low_candidate_class_mg_l)]
        aqua_ref = MICROALBUMIN_AQUA_CONFIRM_REFS[int(aqua_candidate_class_mg_l)]
        return {
            "guard_name": "microalbumin_shade_sanity_check",
            "guard_applied": bool(guard_applied_value),
            "action": str(action_value),
            "original_albumin_value": original_value,
            "corrected_albumin_value": corrected_value,
            "low_candidate_class_mg_l": int(low_candidate_class_mg_l),
            "low_candidate_rgb": tuple(int(x) for x in low_ref["rgb"]),
            "low_candidate_hex": str(low_ref["hex"]),
            "low_candidate_visual_name": str(low_ref["visual_name"]),
            "aqua_candidate_class_mg_l": int(aqua_candidate_class_mg_l),
            "aqua_candidate_rgb": tuple(int(x) for x in aqua_ref["rgb"]),
            "aqua_candidate_hex": str(aqua_ref["hex"]),
            "aqua_candidate_visual_name": str(aqua_ref["visual_name"]),
            "median_de_10": float(median_de_10), "median_de_30": float(median_de_30),
            "median_de_80": float(median_de_80), "median_de_150": float(median_de_150),
            "median_low_de": float(median_low_de), "median_aqua_de": float(median_aqua_de),
            "low_margin": float(low_margin), "aqua_margin": float(aqua_margin),
            "low_pixel_fraction": float(low_pixel_fraction), "aqua_pixel_fraction": float(aqua_pixel_fraction),
            "high_value_confirmed_by_aqua": bool(high_value_confirmed_by_aqua),
            "confidence_bucket": str(confidence_bucket),
            "guard_reason": str(guard_reason_value),
        }

    if img_rgb_uint8 is None:
        action = "unchanged_missing_image"
        guard_reason = "Missing white-balanced image for shade sanity check."
        return _build_response(action, guard_reason, current_albumin_value, corrected_albumin_value)

    if pod_mask_bool is None or current_albumin_value is None:
        action = "unchanged_missing_mask" if pod_mask_bool is None else "unchanged_missing_current_value"
        guard_reason = "Missing microalbumin pod mask or current value for shade sanity check."
        return _build_response(action, guard_reason, current_albumin_value, corrected_albumin_value)

    original_value = current_albumin_value
    pod_eroded = eroded_mask(pod_mask_bool)
    total_pixels = int(pod_eroded.sum())

    if total_pixels < MICRO_SHADE_MIN_PIXELS:
        return _build_response(
            "unchanged_insufficient_mask",
            "Insufficient microalbumin pod pixels for shade sanity check.",
            original_value,
            original_value,
        )

    rgb_pixels = img_rgb_uint8[pod_eroded]
    lab_pixels = rgb2lab(rgb_pixels.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float64)

    de_10 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_LOW_SHADE_REFS_LAB[10])
    de_30 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_LOW_SHADE_REFS_LAB[30])
    de_80 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_AQUA_CONFIRM_REFS_LAB[80])
    de_150 = _deltae_pixels_to_lab_ref(lab_pixels, MICROALBUMIN_AQUA_CONFIRM_REFS_LAB[150])

    low_pixel_de = np.minimum(de_10, de_30)
    aqua_pixel_de = np.minimum(de_80, de_150)

    median_de_10 = float(np.median(de_10))
    median_de_30 = float(np.median(de_30))
    median_de_80 = float(np.median(de_80))
    median_de_150 = float(np.median(de_150))
    median_low_de = min(median_de_10, median_de_30)
    median_aqua_de = min(median_de_80, median_de_150)

    low_candidate_class_mg_l = 10 if median_de_10 <= median_de_30 else 30
    aqua_candidate_class_mg_l = 80 if median_de_80 <= median_de_150 else 150
    low_margin = abs(median_de_10 - median_de_30)
    aqua_margin = abs(median_de_80 - median_de_150)

    low_evidence_mask = (
        (low_pixel_de <= MICRO_SHADE_PIXEL_DE_ACCEPT) &
        (low_pixel_de + MICRO_SHADE_ADVANTAGE_MARGIN < aqua_pixel_de)
    )
    aqua_evidence_mask = (
        (aqua_pixel_de <= MICRO_SHADE_PIXEL_DE_ACCEPT) &
        (aqua_pixel_de + MICRO_SHADE_ADVANTAGE_MARGIN < low_pixel_de)
    )
    low_pixel_fraction = float(low_evidence_mask.sum() / max(total_pixels, 1))
    aqua_pixel_fraction = float(aqua_evidence_mask.sum() / max(total_pixels, 1))

    current_value_float = None if current_albumin_value is None else float(current_albumin_value)
    current_is_high = current_value_float is not None and current_value_float > 30.0
    high_value_confirmed_by_aqua = bool(
        current_is_high and (
            aqua_pixel_fraction >= MICRO_AQUA_SHADE_FRACTION_MIN or
            (
                median_aqua_de <= MICRO_SHADE_MEDIAN_LOW_DE_ACCEPT and
                median_aqua_de + MICRO_SHADE_ADVANTAGE_MARGIN < median_low_de
            )
        )
    )

    guard_applied = False
    action = "unchanged_high_ambiguous_not_low_enough"
    guard_reason = (
        "Higher albumin value lacks strong aqua-blue confirmation, but the pod also lacks "
        "sufficient 10/30 mg/L low-shade evidence for safe downward correction."
    )
    corrected_albumin_value = current_value_float if current_value_float is not None else current_albumin_value

    # Conservative safety guard: never assign high values from these references.
    # The 80/150 aqua references are used only to confirm or veto an already-high label.
    # This only vetoes likely false upward labels and does not diagnose kidney disease.
    # Clinical interpretation still depends on ACR, not albumin colour alone.
    if current_is_high:
        if (
            (not high_value_confirmed_by_aqua) and
            (median_low_de <= MICRO_SHADE_MEDIAN_LOW_DE_ACCEPT) and
            (low_pixel_fraction >= MICRO_LOW_SHADE_FRACTION_MIN) and
            (aqua_pixel_fraction < MICRO_AQUA_SHADE_FRACTION_MIN)
        ):
            corrected_albumin_value = float(low_candidate_class_mg_l)
            guard_applied = True
            action = "override_high_to_low_nearest_shade"
            guard_reason = (
                "Higher albumin label was not supported by 80/150 mg/L aqua-blue evidence; "
                "pod is nearest to the 10/30 mg/L low-albumin shade group."
            )
        elif high_value_confirmed_by_aqua:
            action = "unchanged_high_confirmed_by_aqua"
            guard_reason = "Higher albumin value has aqua-blue visual evidence; low-end override blocked."
    else:
        strong_low_evidence = (
            median_low_de <= MICRO_SHADE_MEDIAN_LOW_DE_ACCEPT and
            low_pixel_fraction >= MICRO_LOW_SHADE_FRACTION_MIN
        )
        if strong_low_evidence:
            corrected_albumin_value = float(low_candidate_class_mg_l)
            action = "confirmed_low_nearest_shade"
            guard_reason = "Existing albumin value is already low; nearest 10/30 mg/L shade confirmed."
        else:
            action = "unchanged_low_value_not_visually_confirmed"
            guard_reason = "Existing albumin value is low, but low-shade evidence was not strong enough for visual refinement."

    if guard_applied:
        confidence_bucket = "High" if (low_pixel_fraction >= 0.70 and aqua_pixel_fraction < 0.02) else "Moderate"
    elif action == "confirmed_low_nearest_shade":
        confidence_bucket = "Moderate"
    elif high_value_confirmed_by_aqua:
        confidence_bucket = "Informational"
    else:
        confidence_bucket = "Low"

    if low_margin < 1.5 and corrected_albumin_value in (10, 10.0, 30, 30.0):
        if confidence_bucket == "High":
            confidence_bucket = "Moderate"
        guard_reason += " The 10 and 30 mg/L reference shades are visually close; nearest shade selected by median ΔE."

    if guard_applied:
        logging.warning(
            "Microalbumin shade guard corrected high albumin value %s to %s. "
            "low_fraction=%.3f aqua_fraction=%.3f reason=%s",
            original_value,
            corrected_albumin_value,
            low_pixel_fraction,
            aqua_pixel_fraction,
            guard_reason,
        )

    return _build_response(
        action, guard_reason, original_value, corrected_albumin_value,
        low_candidate_class_mg_l, aqua_candidate_class_mg_l,
        median_de_10, median_de_30, median_de_80, median_de_150,
        median_low_de, median_aqua_de, low_margin, aqua_margin,
        low_pixel_fraction, aqua_pixel_fraction, high_value_confirmed_by_aqua,
        confidence_bucket, guard_applied,
    )

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
    if val is None: return '1800' if pod_type == 'pod2' else '300 (4.0)'
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
                          uacr_confidence=None):
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    axs[0].imshow(raw_img, interpolation='nearest'); axs[0].axis('off'); axs[0].set_title('Original')

    patch1 = np.ones((50, 50, 3), np.uint8) * p1_mean_display.reshape(1, 1, 3)
    disp1 = find_closest_reference_value_label(calibrated_p1, 'pod1')
    p1_quality_suffix = ""
    if pod_quality and pod_quality.get('creatinine'):
        q = pod_quality['creatinine']
        p1_quality_suffix = f"\nConf {q.get('confidence_pct', 'NA')}% ({q.get('confidence_bucket', 'NA')})"
    axs[1].imshow(patch1, interpolation='nearest'); axs[1].axis('off'); axs[1].set_title(f"Creatinine\n{disp1} mg/dL\nRGB{tuple(p1_mean_display)}{p1_quality_suffix}")

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
            p2_guard_suffix = (
                f"\nShade guard: {action}"
                f"\nLow evidence: {low_pct:.1f}%"
                f"\nAqua evidence: {aqua_pct:.1f}%"
                f"\nLow visual: {low_visual}"
            )
    axs[2].imshow(patch2, interpolation='nearest'); axs[2].axis('off'); axs[2].set_title(f"Microalbumin\n{disp2} mg/L\nRGB{tuple(p2_mean_display)}{p2_quality_suffix}{p2_guard_suffix}")

    if uacr_display:
        color = 'darkgreen' if 'A1' in uacr_display else '#D98E04' if 'A2' in uacr_display else 'darkred'
        conf_text = f"\nEvidence confidence: {uacr_confidence}%" if uacr_confidence is not None else ""
        fig.suptitle(f"UACR Result\n{uacr_display}{conf_text}", fontsize=11, fontweight='bold', color=color, y=0.97)

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
        return None, "Indeterminate", "Unavailable", "Indeterminate: One or both values could not be calculated."

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
    )
    c2_final = albumin_shade_guard.get("corrected_albumin_value", c2_snapped)
    if c2_final is None:
        c2_final = c2_snapped

    # Preserve legacy behavior trace (continuous calibrated values) and corrected
    # behavior trace (snapped/displayed values) to avoid disruption in existing flow.
    uacr_legacy_value, _, _, _ = calculate_uacr_and_category(c2, c1)
    uacr_value, uacr_stage, uacr_range, uacr_display = calculate_uacr_and_category(c2_final, c1_snapped)
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
        c2_final,
        uacr_display,
        fpath,
        pod_quality={
            'creatinine': pod1_conf,
            'microalbumin': {
                **pod2_conf,
                'shade_sanity_check': albumin_shade_guard,
            },
        },
        uacr_confidence=uacr_confidence,
    )
    
    return {
        'composite_img': fname,
        # Corrected value aligned to displayed snapped analyte values.
        'uacr_value': uacr_value,
        'uacr_category': uacr_display,
        'uacr_stage': uacr_stage,
        'uacr_reference_range': uacr_range,
        # Traceability fields to preserve legacy vs corrected outputs.
        'uacr_legacy_value': uacr_legacy_value,
        'uacr_corrected_value': uacr_value,
        'uacr_delta_legacy_minus_corrected': uacr_delta,
        'uacr_legacy_trace': build_uacr_trace(c2, c1, uacr_legacy_value, 'legacy_calibrated_continuous'),
        'uacr_corrected_trace': build_uacr_trace(
            c2_final,
            c1_snapped,
            uacr_value,
            'corrected_snapped_displayed_with_microalbumin_shade_guard'
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
                'final_display_value': c2_final,
                'low_end_snap_label': c2_low_end_label,
                'shade_sanity_check': albumin_shade_guard,
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
