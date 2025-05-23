import os
import sys
import torch
import numpy as np
from flask import Blueprint, render_template, request
from PIL import Image
from .utils import process_image_and_get_pods

# ───────────────────────────────
# Create Flask Blueprint
# ───────────────────────────────
# This allows modular separation of routes from the main app
main = Blueprint("main", __name__)

# ───────────────────────────────
# Utility: Handle PyInstaller Executable Paths
# ───────────────────────────────
def resource_path(relative_path):
    """
    Returns the absolute path to a resource (e.g. model file),
    compatible with both local development and packaged .exe (PyInstaller).
    """
    try:
        # When packaged with PyInstaller, sys._MEIPASS points to the temp folder
        base_path = sys._MEIPASS
    except AttributeError:
        # In development mode, use current working directory
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ───────────────────────────────
# Load TorchScript Model
# ───────────────────────────────
# Define model path using resource_path helper
MODEL_PATH = resource_path("app/model/pod_segmentation_scriptedV3.pt")

# Automatically select GPU if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the serialized TorchScript model
model = torch.jit.load(MODEL_PATH, map_location=DEVICE).to(DEVICE).eval()

# ───────────────────────────────
# Main Web Route: Upload & Inference
# ───────────────────────────────
@main.route("/", methods=["GET", "POST"])
def index():
    """
    Handles both GET and POST requests:
    - GET: Renders the homepage with upload form
    - POST: Receives uploaded image, runs model inference,
            and returns composite visualization results
    """
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file:
            # Save uploaded file to static/uploads/
            upload_path = os.path.join("app", "static", "uploads", image_file.filename)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            image_file.save(upload_path)

            # Run model on the uploaded image and generate summary output
            result = process_image_and_get_pods(upload_path, model, DEVICE)

            # Render result view with image and output details
            return render_template("index.html", result=result, image_name=image_file.filename)

    # If GET request, simply load upload interface
    return render_template("index.html")
