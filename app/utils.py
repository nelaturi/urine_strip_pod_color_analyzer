import matplotlib
matplotlib.use('Agg') # This line MUST be before 'import matplotlib.pyplot as plt'

import os
import cv2
import uuid
import sys
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt # This import should come AFTER matplotlib.use('Agg')
import joblib
from skimage.color import rgb2lab


# ───────────────────────────────
# Constants & Configuration
# ───────────────────────────────

# Class indices corresponding to the segmentation model output
# 4 = Pod1 (Creatinine), 5 = Pod2 (Microalbumin)
POD1_IDX, POD2_IDX = 4, 5

# Albumentations preprocessing pipeline for inference
val_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ───────────────────────────────
# PyInstaller Compatibility
# ───────────────────────────────

def resource_path(relative_path):
    """
    Get absolute path to resource.
    Works both in development and in PyInstaller-packed executables.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ───────────────────────────────
# Load Calibrated Models
# ───────────────────────────────

CREATININE_MODEL_PATH = resource_path("app/model/creatinine_model.pkl")
MICROALBUMIN_MODEL_PATH = resource_path("app/model/microalbumin_model.pkl")

model_creat = None
model_micro = None

try:
    model_creat = joblib.load(CREATININE_MODEL_PATH)
    model_micro = joblib.load(MICROALBUMIN_MODEL_PATH)
    print(f"[{os.path.basename(__file__)}] Calibrated models (Creatinine, Microalbumin) loaded successfully.")
except Exception as e:
    print(f"[{os.path.basename(__file__)}] ERROR loading calibrated models: {e}")
    print(f"[{os.path.basename(__file__)}] Please ensure '{CREATININE_MODEL_PATH}' and '{MICROALBUMIN_MODEL_PATH}' exist.")


POD_COLOR_CHART = {
    "pod2": {  # Microalbumin
        "10": (224, 255, 224),     # very light green
        "30": (204, 255, 153),     # light green-yellow
        "80": (255, 255, 102),     # bright yellow
        "150": (255, 204, 0),      # golden yellow
    },
    "pod1": {  # Creatinine
        "10 (0.1)": (255, 204, 153),   # pale orange
        "50 (0.5)": (255, 153, 51),    # orange
        "100 (1.0)": (204, 102, 0),    # burnt orange
        "200 (2.0)": (153, 76, 0),     # brownish orange
        "300 (4.0)": (128, 64, 0),     # dark brown
    }
}

COLOR_NAMES = {
    "10": "very light green",
    "30": "light yellow green",
    "80": "bright yellow",
    "150": "golden yellow",
    "10 (0.1)": "pale orange",
    "50 (0.5)": "vivid orange",
    "100 (1.0)": "burnt orange",
    "200 (2.0)": "brown orange",
    "300 (4.0)": "dark brown"
}

REFERENCE_VALUES = {
    "pod1": {
        "10 (0.1)": 10,    # Creatinine in mg/dL
        "50 (0.5)": 50,
        "100 (1.0)": 100,
        "200 (2.0)": 200,
        "300 (4.0)": 300,
    },
    "pod2": {
        "10": 10,    # Microalbumin in mg/g or μg/mL
        "30": 30,
        "80": 80,
        "150": 150,
    }
}

# ───────────────────────────────
# Utility: Extract Features for Calibration Models
# ───────────────────────────────

def extract_features_from_rgb(rgb_triplet):
    """
    Extracts RGB, HSV, and Lab features from an RGB color triplet.
    Input rgb_triplet should be a tuple/list of 3 integers (0-255).
    """
    rgb_array = np.uint8([[rgb_triplet]])
    lab = rgb2lab(rgb_array).reshape(3,)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).reshape(3,)
    return list(rgb_triplet) + list(hsv) + list(lab)

def get_calibrated_value(rgb_triplet, pod_type):
    """
    Uses the calibrated models to predict the clinical value for a given RGB triplet.
    Returns None if models are not loaded or pod_type is invalid.
    """
    if model_creat is None or model_micro is None:
        print(f"[{os.path.basename(__file__)}] WARNING: Calibrated models not loaded, returning None for {pod_type}.")
        return None

    features = extract_features_from_rgb(rgb_triplet)

    try:
        if pod_type == "creatinine":
            return float(model_creat.predict([features])[0])
        elif pod_type == "microalbumin":
            return float(model_micro.predict([features])[0])
        else:
            print(f"[{os.path.basename(__file__)}] WARNING: Unknown pod_type '{pod_type}'. No calibrated value available.")
            return None
    except Exception as e:
        print(f"[{os.path.basename(__file__)}] ERROR predicting with calibrated model for {pod_type}: {e}")
        return None


# ───────────────────────────────
# Utility: Find Closest Color Match (now purely for color name, not primary value)
# ───────────────────────────────

def match_patch_color(pod_rgb, pod_type):
    """
    Given the mean RGB of a pod region, find the closest matching
    clinical label and color name based on the reference chart.
    This is now primarily for generating a human-readable color name for the image text.
    """
    ref_colors = POD_COLOR_CHART[pod_type]
    pod_rgb = np.array(pod_rgb).astype(np.float32)

    min_dist = float("inf")
    closest_label_by_rgb = "Unknown"

    for label, ref_rgb in ref_colors.items():
        ref_rgb = np.array(ref_rgb).astype(np.float32)
        dist = np.linalg.norm(pod_rgb - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_label_by_rgb = label

    color_name = COLOR_NAMES.get(closest_label_by_rgb, "Unknown color")
    return closest_label_by_rgb, color_name # Return original label and color name


# ───────────────────────────────
# Utility: Find Closest Reference Chart Value
# ───────────────────────────────
def find_closest_reference_value_label(calibrated_value, pod_type):
    """
    Given a calibrated numerical value, find the closest label from the
    predefined REFERENCE_VALUES chart for the given pod type.
    """
    if calibrated_value is None:
        return "N/A"

    ref_chart = REFERENCE_VALUES.get(pod_type)
    if not ref_chart:
        return f"{calibrated_value:.2f}" # Fallback if no reference chart defined

    min_diff = float('inf')
    closest_label = None

    for label, ref_value in ref_chart.items():
        diff = abs(calibrated_value - ref_value)
        if diff < min_diff:
            min_diff = diff
            closest_label = label
    
    return closest_label if closest_label is not None else f"{calibrated_value:.2f}" # Fallback if no closest found


# ───────────────────────────────
# Generate Annotated Visual Output (MODIFIED FOR RAW IMAGE DISPLAY)
# ───────────────────────────────

def save_composite_visual(raw_img, pod1_region, pod2_region, p1_mean, p2_mean,
                          calibrated_p1_value, calibrated_p2_value, save_path):
    """
    Save a visual summary image containing:
    - Original uploaded image
    - Color patch for each pod
    - RGB values
    - Calibrated clinical label (or original if calibration fails)
    - Color name
    """
    print(f"[{os.path.basename(__file__)}] Starting save_composite_visual...")

    # Define figure and subplots: 1 row, 3 columns
    # Adjust figsize to accommodate the third image while maintaining aspect ratio
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 5)) # Increased width from 6.5 to 9.5

    # --- Plot Original Image ---
    axs[0].imshow(raw_img)
    axs[0].set_title("Original Image", fontsize=11)
    axs[0].axis("off")
    # Ensure aspect ratio is preserved and image fills the subplot height
    axs[0].set_aspect('auto') # 'auto' will stretch/squish to fill axes, 'equal' keeps aspect.
                            # 'auto' often works better for mixed content. If you want strict aspect,
                            # you might need to pad the image or adjust figure dimensions carefully.
                            # Given "reduce image size... but appearance should not change", 'auto' is flexible.
                            # For fixed aspect, set: axs[0].set_aspect('equal', adjustable='box')


    # Generate color patches from average RGB
    patch_size = 70
    patch1 = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * p1_mean.reshape(1, 1, 3)
    patch2 = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * p2_mean.reshape(1, 1, 3)

    # Get human-readable color names (still based on original chart for description)
    _, colorname1 = match_patch_color(p1_mean, "pod1")
    _, colorname2 = match_patch_color(p2_mean, "pod2")

    # Determine the value to display for Creatinine - NOW SNAPPED TO REFERENCE LABEL
    display_value_p1 = find_closest_reference_value_label(calibrated_p1_value, "pod1")

    # Determine the value to display for Microalbumin - NOW SNAPPED TO REFERENCE LABEL
    display_value_p2 = find_closest_reference_value_label(calibrated_p2_value, "pod2")


    # --- Plot Pod1 = Creatinine (now in axs[1]) ---
    axs[1].imshow(patch1)
    axs[1].axis("off")
    axs[1].text(0.5, 1.3, "Creatinine", fontsize=11, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, 1.18, f"Color: {colorname1}", fontsize=10, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, 1.05, f"Value: {display_value_p1}", fontsize=10, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, -0.25, f"RGB ({p1_mean[0]}, {p1_mean[1]}, {p1_mean[2]})",
                fontsize=10, ha="center", va="top", transform=axs[1].transAxes)

    # --- Plot Pod2 = Microalbumin (now in axs[2]) ---
    axs[2].imshow(patch2)
    axs[2].axis("off")
    axs[2].text(0.5, 1.3, "Microalbumin", fontsize=11, ha="center", transform=axs[2].transAxes)
    axs[2].text(0.5, 1.18, f"Color: {colorname2}", fontsize=10, ha="center", transform=axs[2].transAxes)
    axs[2].text(0.5, 1.05, f"Value: {display_value_p2}", fontsize=10, ha="center", transform=axs[2].transAxes)
    axs[2].text(0.5, -0.25, f"RGB ({p2_mean[0]}, {p2_mean[1]}, {p2_mean[2]})",
                fontsize=10, ha="center", va="top", transform=axs[2].transAxes)

    plt.subplots_adjust(top=0.9, bottom=0.25)
    
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[{os.path.basename(__file__)}] Composite image saved to: {save_path}")
    except Exception as e:
        print(f"[{os.path.basename(__file__)}] ERROR saving composite image to {save_path}: {e}")
    finally:
        plt.close(fig) # Always close the figure to free up memory

# ───────────────────────────────
# Main Inference Logic (no change required here beyond passing raw_img)
# ───────────────────────────────

def process_image_and_get_pods(image_path, model, device):
    """
    Full image processing pipeline:
    - Load uploaded image
    - Preprocess and infer using segmentation model
    - Identify pod masks
    - Compute RGB means
    - Apply calibrated models to get refined clinical values
    - Generate visual summary (with calibrated values) and return composite image filename
    """
    print(f"[{os.path.basename(__file__)}] Starting process_image_and_get_pods for {image_path}")

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    unique_id = uuid.uuid4().hex[:6]

    try:
        img_pil = Image.open(image_path).convert("RGB")
        if img_pil.width > img_pil.height:
            img_pil = img_pil.rotate(90, expand=True)
        img_np = np.array(img_pil) # This is the raw image numpy array
        original_size = img_np.shape[:2]
        print(f"[{os.path.basename(__file__)}] Image loaded and dimensions: {original_size}")
    except Exception as e:
        print(f"[{os.path.basename(__file__)}] ERROR: Could not load or process image {image_path}: {e}")
        raise # Re-raise to be caught by routes.py

    try:
        model_input = val_tf(image=img_np)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(model_input)
            preds = logits.argmax(dim=1).cpu().numpy()[0]
        print(f"[{os.path.basename(__file__)}] Model inference completed.")
    except Exception as e:
        print(f"[{os.path.basename(__file__)}] ERROR: Model inference failed: {e}")
        raise

    mask_resized = cv2.resize(preds.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Initialize regions as black if no pixels are found
    pod1_region = np.zeros_like(img_np)
    pod2_region = np.zeros_like(img_np)

    pod1_mask = (mask_resized == POD1_IDX)
    pod2_mask = (mask_resized == POD2_IDX)

    if pod1_mask.any():
        pod1_region = np.where(pod1_mask[..., None], img_np, 0)
        p1_pix = pod1_region[pod1_mask].astype(np.float32)
        p1_mean = np.round(p1_pix.mean(axis=0)).astype(np.uint8)
        print(f"[{os.path.basename(__file__)}] Pod1 (Creatinine) detected. Mean RGB: {p1_mean}")
    else:
        p1_mean = np.array([0, 0, 0], dtype=np.uint8)
        print(f"[{os.path.basename(__file__)}] Pod1 (Creatinine) NOT detected. Defaulting to black RGB.")


    if pod2_mask.any():
        pod2_region = np.where(pod2_mask[..., None], img_np, 0)
        p2_pix = pod2_region[pod2_mask].astype(np.float32)
        p2_mean = np.round(p2_pix.mean(axis=0)).astype(np.uint8)
        print(f"[{os.path.basename(__file__)}] Pod2 (Microalbumin) detected. Mean RGB: {p2_mean}")
    else:
        p2_mean = np.array([0, 0, 0], dtype=np.uint8)
        print(f"[{os.path.basename(__file__)}] Pod2 (Microalbumin) NOT detected. Defaulting to black RGB.")


    # --- Get calibrated clinical values ---
    calibrated_creatinine_value = get_calibrated_value(tuple(p1_mean), "creatinine")
    calibrated_microalbumin_value = get_calibrated_value(tuple(p2_mean), "microalbumin")
    print(f"[{os.path.basename(__file__)}] Calibrated Creatinine: {calibrated_creatinine_value}, Microalbumin: {calibrated_microalbumin_value}")
    # --- END NEW ---

    output_folder = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(output_folder, exist_ok=True)

    composite_filename = f"{base_filename}_visual_{unique_id}.png"
    composite_path = os.path.join(output_folder, composite_filename)

    # --- Pass raw_img and calibrated values to save_composite_visual ---
    try:
        save_composite_visual(img_np, pod1_region, pod2_region, p1_mean, p2_mean,
                              calibrated_creatinine_value, calibrated_microalbumin_value,
                              composite_path)
    except Exception as e:
        print(f"[{os.path.basename(__file__)}] ERROR in save_composite_visual: {e}")
        return {"composite_img": None}


    return {
        "composite_img": composite_filename
    }
