import os
import cv2
import uuid
import sys
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

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
# Reference Color Charts
# ───────────────────────────────

# Average RGB values for each pod label (clinically mapped)
POD_COLOR_CHART = {
    "pod2": {  # Microalbumin
        "10": (151, 181, 155),
        "30": (159, 195, 167),
        "80": (153, 211, 199),
        "150": (123, 201, 205),
    },
    "pod1": {  # Creatinine
        "10 (0.1)": (162, 121, 3),
        "50 (0.5)": (168, 151, 0),
        "100 (1.0)": (182, 170, 6),
        "200 (2.0)": (166, 162, 29),
        "300 (4.0)": (144, 131, 61),
    }
}

# Human-readable color names corresponding to RGB reference chart
COLOR_NAMES = {
    "10": "very light aqua",
    "30": "light aqua",
    "80": "pale cyan",
    "150": "sky blue",
    "10 (0.1)": "dark amber",
    "50 (0.5)": "mustard yellow",
    "100 (1.0)": "golden olive",
    "200 (2.0)": "greenish yellow",
    "300 (4.0)": "olive brown"
}

# ───────────────────────────────
# Utility: Find Closest Color Match
# ───────────────────────────────

def match_patch_color(pod_rgb, pod_type):
    """
    Given the mean RGB of a pod region, find the closest matching
    clinical label and color name based on the reference chart.
    """
    ref_colors = POD_COLOR_CHART[pod_type]
    pod_rgb = np.array(pod_rgb).astype(np.float32)

    min_dist = float("inf")
    closest_label = "Unknown"

    for label, ref_rgb in ref_colors.items():
        ref_rgb = np.array(ref_rgb).astype(np.float32)
        dist = np.linalg.norm(pod_rgb - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_label = label

    color_name = COLOR_NAMES.get(closest_label, "Unknown color")
    return closest_label, color_name

# ───────────────────────────────
# Generate Annotated Visual Output
# ───────────────────────────────

def save_composite_visual(raw_img, pod1_img, pod2_img, p1_mean, p2_mean, save_path):
    """
    Save a visual summary image containing:
    - Color patch for each pod
    - RGB values
    - Clinical label
    - Color name
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

    # Generate color patches from average RGB
    patch_size = 70
    patch1 = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * p1_mean.reshape(1, 1, 3)
    patch2 = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * p2_mean.reshape(1, 1, 3)

    # Match patch colors to known labels
    label1, colorname1 = match_patch_color(p1_mean, "pod1")
    label2, colorname2 = match_patch_color(p2_mean, "pod2")

    # Left: Pod1 = Creatinine
    axs[0].imshow(patch1)
    axs[0].axis("off")
    axs[0].text(0.5, 1.3, "Creatinine", fontsize=11, ha="center", transform=axs[0].transAxes)
    axs[0].text(0.5, 1.18, f"Color: {colorname1}", fontsize=10, ha="center", transform=axs[0].transAxes)
    axs[0].text(0.5, 1.05, f"Value: {label1}", fontsize=10, ha="center", transform=axs[0].transAxes)
    axs[0].text(0.5, -0.25, f"RGB ({p1_mean[0]}, {p1_mean[1]}, {p1_mean[2]})",
                fontsize=10, ha="center", va="top", transform=axs[0].transAxes)

    # Right: Pod2 = Microalbumin
    axs[1].imshow(patch2)
    axs[1].axis("off")
    axs[1].text(0.5, 1.3, "Microalbumin", fontsize=11, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, 1.18, f"Color: {colorname2}", fontsize=10, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, 1.05, f"Value: {label2}", fontsize=10, ha="center", transform=axs[1].transAxes)
    axs[1].text(0.5, -0.25, f"RGB ({p2_mean[0]}, {p2_mean[1]}, {p2_mean[2]})",
                fontsize=10, ha="center", va="top", transform=axs[1].transAxes)

    plt.subplots_adjust(top=0.9, bottom=0.25)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ───────────────────────────────
# Main Inference Logic
# ───────────────────────────────

def process_image_and_get_pods(image_path, model, device):
    """
    Full image processing pipeline:
    - Load uploaded image
    - Preprocess and infer using segmentation model
    - Identify pod masks
    - Compute RGB means
    - Generate visual summary and return RGB + file path
    """

    # Generate unique filename for saving outputs
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    unique_id = uuid.uuid4().hex[:6]

    # Load and auto-rotate vertical images
    img_pil = Image.open(image_path).convert("RGB")
    if img_pil.width > img_pil.height:
        img_pil = img_pil.rotate(90, expand=True)

    img_np = np.array(img_pil)
    original_size = img_np.shape[:2]

    # Apply transforms and send to model
    model_input = val_tf(image=img_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(model_input)
        preds = logits.argmax(dim=1).cpu().numpy()[0]

    # Resize prediction back to original size
    mask_resized = cv2.resize(preds.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # Create pod masks
    pod1_mask = (mask_resized == POD1_IDX)
    pod2_mask = (mask_resized == POD2_IDX)

    # Extract pod-specific regions
    pod1_region = np.where(pod1_mask[..., None], img_np, 0)
    pod2_region = np.where(pod2_mask[..., None], img_np, 0)

    # Compute mean RGB values for each pod
    p1_pix = pod1_region[pod1_mask].astype(np.float32)
    p2_pix = pod2_region[pod2_mask].astype(np.float32)

    p1_mean = np.round(p1_pix.mean(axis=0)).astype(np.uint8) if p1_pix.size > 0 else np.array([0, 0, 0], dtype=np.uint8)
    p2_mean = np.round(p2_pix.mean(axis=0)).astype(np.uint8) if p2_pix.size > 0 else np.array([0, 0, 0], dtype=np.uint8)

    # Save composite visual output to uploads folder
    output_folder = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(output_folder, exist_ok=True)

    composite_filename = f"{base_filename}_visual_{unique_id}.png"
    composite_path = os.path.join(output_folder, composite_filename)

    save_composite_visual(img_np, pod1_region, pod2_region, p1_mean, p2_mean, composite_path)

    # Return structured output
    return {
        "p1_rgb": tuple(p1_mean),
        "p2_rgb": tuple(p2_mean),
        "composite_img": composite_filename
    }
