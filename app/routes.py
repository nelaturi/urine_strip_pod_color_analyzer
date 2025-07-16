import os
import sys
import torch
import numpy as np
from flask import Blueprint, render_template, request, url_for
import uuid
from .utils import process_image_and_get_pods

# ───────────────────────────────
# Create Flask Blueprint
# ───────────────────────────────
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
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ───────────────────────────────
# Load TorchScript Model
# ───────────────────────────────
MODEL_PATH = resource_path("app/model/pod_segmentation_scriptedV3.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE).to(DEVICE).eval()
    print(f"[{os.path.basename(__file__)}] Main segmentation model loaded successfully from {MODEL_PATH} on {DEVICE}.")
except Exception as e:
    print(f"[{os.path.basename(__file__)}] ERROR loading main segmentation model: {e}")
    print(f"[{os.path.basename(__file__)}] Please ensure '{MODEL_PATH}' exists and is a valid TorchScript model.")

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
    result_for_template = None
    image_name_for_template = None
    error_message = None

    print(f"[{os.path.basename(__file__)}] Request method: {request.method}")

    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file and image_file.filename != '':
            print(f"[{os.path.basename(__file__)}] Image file received: {image_file.filename}")
            if model is None:
                error_message = "Segmentation model failed to load. Cannot process image."
                print(f"[{os.path.basename(__file__)}] ERROR: Segmentation model is None.")
            else:
                original_filename = image_file.filename
                
                upload_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                
                temp_upload_path = os.path.join(upload_dir, f"temp_{uuid.uuid4().hex}_{original_filename}")
                image_file.save(temp_upload_path)
                print(f"[{os.path.basename(__file__)}] Image saved temporarily to: {temp_upload_path}")

                try:
                    print(f"[{os.path.basename(__file__)}] Calling process_image_and_get_pods...")
                    result_data = process_image_and_get_pods(temp_upload_path, model, DEVICE)
                    print(f"[{os.path.basename(__file__)}] process_image_and_get_pods returned: {result_data}")

                    if result_data and result_data.get('composite_img'):
                        result_for_template = result_data
                        image_name_for_template = result_data['composite_img']
                    else:
                        error_message = "Image processing completed but no composite image was generated."
                        print(f"[{os.path.basename(__file__)}] WARNING: No composite image returned from process_image_and_get_pods.")

                except Exception as e:
                    error_message = f"Error processing image: {e}"
                    print(f"[{os.path.basename(__file__)}] ERROR during image processing: {e}")
                finally:
                    if os.path.exists(temp_upload_path):
                        os.remove(temp_upload_path)
                        print(f"[{os.path.basename(__file__)}] Removed temporary image: {temp_upload_path}")
        else:
            error_message = "No image file provided or file is empty."
            print(f"[{os.path.basename(__file__)}] WARNING: No image file or empty file received.")

    print(f"[{os.path.basename(__file__)}] Rendering index.html. image_name_for_template: {image_name_for_template}, Result dict exists: {result_for_template is not None}, Error: {error_message is not None}")
    return render_template("index.html",
                           result=result_for_template,
                           image_name=image_name_for_template,
                           error_message=error_message)