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
                          p1_mean_raw, p2_mean_raw,
                          calibrated_p1, calibrated_p2,
                          uacr_display,
                          save_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    axs[0].imshow(raw_img); axs[0].axis('off'); axs[0].set_title('Original')

    patch1 = np.ones((50, 50, 3), np.uint8) * p1_mean_raw.reshape(1, 1, 3)
    disp1 = find_closest_reference_value_label(calibrated_p1, 'pod1')
    axs[1].imshow(patch1); axs[1].axis('off'); axs[1].set_title(f"Creatinine\n{disp1} mg/dL\nRGB{tuple(p1_mean_raw)}")

    patch2 = np.ones((50, 50, 3), np.uint8) * p2_mean_raw.reshape(1, 1, 3)
    lab_obs = _rgb_to_lab_triplet(tuple(p2_mean_raw))
    color_lbl, color_de = nearest_micro_centroid_lab(lab_obs)
    reg_lbl = find_closest_reference_value_label(calibrated_p2, 'pod2')

    if reg_lbl.isdigit() and int(reg_lbl) in MICROALBUMIN_CENTROIDS_LAB:
        lab_reg_cent = MICROALBUMIN_CENTROIDS_LAB[int(reg_lbl)]
        reg_de = _lab_distance(lab_obs, lab_reg_cent)
    else:
        reg_de = float('inf')

    disp2 = color_lbl if (color_de <= MICRO_DELTAE_ACCEPT) and ((reg_de - color_de) > MICRO_DELTAE_MARGIN) else reg_lbl
    axs[2].imshow(patch2); axs[2].axis('off'); axs[2].set_title(f"Microalbumin\n{disp2} mg/L\nRGB{tuple(p2_mean_raw)}")

    if uacr_display:
        color = 'darkgreen' if 'A1' in uacr_display else '#D98E04' if 'A2' in uacr_display else 'darkred'
        fig.suptitle(f"UACR Result\n{uacr_display}", fontsize=11, fontweight='bold', color=color, y=0.97)

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

    c1 = get_calibrated_value(tuple(p1_mean_ui), 'creatinine')
    c2 = get_calibrated_value(tuple(p2_mean_ui), 'microalbumin')

    c1_snapped, _ = apply_low_end_snap('creatinine', tuple(p1_mean_ui), c1)
    c2_snapped, _ = apply_low_end_snap('microalbumin', tuple(p2_mean_ui), c2)

    uacr_value, uacr_stage, uacr_range, uacr_display = calculate_uacr_and_category(c2_snapped, c1_snapped)

    out_dir = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{os.path.splitext(os.path.basename(image_path))[0]}_{uuid.uuid4().hex[:6]}.png"
    fpath = os.path.join(out_dir, fname)
    
    save_composite_visual(
        raw_np,
        None,
        None,
        p1_mean_raw,
        p2_mean_raw,
        c1_snapped,
        c2_snapped,
        uacr_display,
        fpath,
    )
    
    return {
        'composite_img': fname,
        'uacr_value': uacr_value,
        'uacr_category': uacr_display,
        'uacr_stage': uacr_stage,
        'uacr_reference_range': uacr_range,
    }
