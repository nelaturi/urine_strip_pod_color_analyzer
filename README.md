
# Urine Strip Pod Color Analyzer

A Flask web application powered by a deep learning model to identify and analyze the color of pods in urine test strips. The system automatically detects Creatinine and Microalbumin pods in uploaded strip images and generates an easy-to-interpret composite summary of pod colors, mapped to clinical color codes.

---

## Features

- Upload a urine test strip image and get instant pod color identification.
- Visual composite summary: Shows color patches, clinical values, and RGB codes for each pod.
- Simple, user-friendly web interface.
- Downloadable results for record-keeping or clinical use.
- TorchScript-powered backend for fast inference.

---

## Folder Structure

```
urine-strip-analyzer/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── utils.py
│   ├── model/
│   │   └── pod_segmentation_scriptedV3.pt  # [Not on GitHub - see instructions]
│   ├── static/
│   │   └── uploads/        # Stores uploaded images & output
│   └── templates/
│       └── index.html
├── run.py
├── requirements.txt
└── README.md
```

---

## **Setup Instructions**

> **Note:** Your trained model file (`pod_segmentation_scriptedV3.pt`) is **over 100MB** and cannot be pushed to GitHub.  
> **Download the model from the link below and place it in `app/model/`:**  
> **[Download model from Google Drive](https://drive.google.com/file/d/1Y_hBEldKNbi-UdaIAgdm8SsfBqNoScO-/view?usp=drive_link)**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/urine-strip-analyzer.git
cd urine-strip-analyzer
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

**(Or, if you use `environment.yml`, run `conda env create -f environment.yml`)**

### 3. Download the Model File

- Download the model file (`pod_segmentation_scriptedV3.pt`) from:  
  **[Download model from Google Drive](https://drive.google.com/file/d/1Y_hBEldKNbi-UdaIAgdm8SsfBqNoScO-/view?usp=drive_link)**
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

- **Model**: TorchScript-serialized segmentation model trained to identify and classify pod regions in urine strip images.
- **Input**: `.jpg` or `.png` images of urine test strips.
- **Output**: Composite summary image with clinical mapping for each pod.

---

## **Notes and Recommendations**

- **Model file not in repo**: Due to GitHub file size limits, the model binary is NOT tracked by Git. Use a cloud link as described.
- **.gitignore**: Add `app/model/pod_segmentation_scriptedV3.pt` and `app/static/uploads/` to your `.gitignore` to avoid pushing large or auto-generated files.

Example `.gitignore`:
```
# Model binary
app/model/pod_segmentation_scriptedV3.pt

# Uploaded & output images
app/static/uploads/
```

- **Production deployment**: For public use, consider using Gunicorn or a cloud host like Render, Heroku, or AWS.
- **License**: Add license terms if required.

---

## **Contact**

For any queries, reach out at [your.email@example.com] or create an issue in this repository.

---

## **Credits**

- Developed by [Your Name]
- Powered by Flask, PyTorch, Albumentations, and Bootstrap.

