
# Urine Strip Pod Color Analyzer (V2)

A Flask web application powered by a deep learning model to identify and analyze the color of pods in urine test strips. The system automatically detects Creatinine and Microalbumin pods in uploaded strip images.

**Key Enhancements in this Version:**
* **Calibrated Clinical Values:** Incorporates specialized regression models for Creatinine and Microalbumin. These models refine the predicted clinical values from the detected pod colors, resulting in significantly higher accuracy compared to direct color-to-value mapping. The displayed values are then intelligently "snapped" to the closest standard value from the clinical reference chart for clarity.
* **Raw Image in Composite Summary:** For enhanced context and visual comparison, the original uploaded urine test strip image is now displayed directly within the composite summary output, alongside the analyzed Creatinine and Microalbumin pods.
* **Improved User Interface:** While maintaining the original minimalist layout, the visual output now provides more precise information directly on the composite image.

The system generates an easy-to-interpret composite summary showing the raw input, detected pod colors, calibrated clinical values, and RGB codes.

## Features

* Upload a urine test strip image and get instant pod color identification.
* Visual composite summary: Shows the **original uploaded image**, color patches for Creatinine and Microalbumin, **calibrated clinical values (snapped to reference labels)**, and RGB codes for each detected pod.
* Simple, user-friendly web interface.
* Downloadable results for record-keeping or clinical use.
* TorchScript-powered segmentation backend for fast inference.
* `joblib`-serialized regression models for accurate clinical value calibration.

## Folder Structure

```
urine_strip_pod_color_analyzer/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── utils.py
│   ├── model/
│   │   └── pod_segmentation_scriptedV3.pt  # [Not on GitHub - see instructions]
│   │   └── creatinine_model.pkl          # [Not on GitHub - see instructions]
│   │   └── microalbumin_model.pkl        # [Not on GitHub - see instructions]
│   ├── static/
│   │   └── uploads/        # Stores uploaded images & Generated Composite Outputs
│   └── templates/
│       └── index.html
├── run.py
├── requirements.txt
└── README.md
```

---

## **Setup Instructions**

> **Download the models from the link below and place it in `app/model/`:**  
> **[Download models from Google Drive] --> https://drive.google.com/drive/folders/1-Gg3kdMehKUKgHVt3BC8C9UD2c6oqciS?usp=sharing**

### 1. Clone the Repository

```bash
git clone https://github.com/nelaturi/urine_strip_pod_color_analyzer_V2.git
cd urine_strip_pod_color_analyzer_V2
```

### 2. Setup Environment (Anaconda Recommended)

Create and activate a new environment:
```bash
conda create -n urine-analyzer python=3.9
conda activate urine-analyzer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download the Model Files

- Download the model files ('pod_segmentation_scriptedV3.pt', 'creatinine_model.pkl', 'microalbumin_model.pkl') from:  
  **[Download model from Google Drive] --> https://drive.google.com/drive/folders/1-Gg3kdMehKUKgHVt3BC8C9UD2c6oqciS?usp=sharing**
- Place it in `app/model/`.

---

## **Usage**

### Run the Flask app:

```bash
python run.py
```

- The app will automatically open in your browser (`http://127.0.0.1:5000`).
- Upload a strip image and view/download the results.

---

## **Project Structure Details**

- **app/**: Main Flask app code
  - **routes.py**: Handles image upload, model inference, and result rendering.
  - **utils.py**: Image preprocessing, segmentation, and color mapping utilities.
  - **model/**: Contains the TorchScript model (not on GitHub).
  - **static/uploads/**: Stores images/results generated during app use.
  - **templates/index.html**: Web interface.

- **run.py**: Launches the Flask app.
- **requirements.txt**: All Python package dependencies.
- **README.md**: You are here.

---

## **Model & Data**

* **Segmentation Model**: A TorchScript-serialized deep learning model, trained to accurately identify and segment the Creatinine and Microalbumin pod regions within urine strip images.
* **Calibrated Prediction Models**: Two separate `joblib`-serialized regression models (one for Creatinine, one for Microalbumin) that take multi-channel color features (RGB, HSV, Lab) extracted from the detected pods as input. These models predict a more accurate, continuous clinical value. For display, these continuous values are then "snapped" to the nearest predefined discrete value on the official reference chart to provide clinically relevant interpretations.
* **Input**: Standard `.jpg` or `.png` images of urine test strips.
* **Output**: A composite summary image visually presenting:
    * The **original uploaded urine strip image**.
    * Color patches representing the average color of the detected Creatinine and Microalbumin pods.
    * The **calibrated clinical value** (mapped to the closest reference chart label).
    * The mean RGB color code for each pod.

---


## **Credits**

- Developed by Naresh Nelaturi
- Powered by Flask, PyTorch, Albumentations, and Bootstrap.
