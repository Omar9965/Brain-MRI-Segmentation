# Brain MRI Tumor Segmentation

A FastAPI + PyTorch application for brain tumor segmentation on MRI scans. It utilizes a state-of-the-art U-Net architecture powered by an EfficientNet-B7 encoder and enhanced with Spatial and Channel Squeeze & Excitation (scSE) attention gates. It provides a simple web UI and a clean, memory-efficient REST API for single image inference, responding with Base64 encoded image data.

---

## Key Features

- **Advanced Architecture:** `U-Net` model with `EfficientNet-B7` encoder and `scSE` attention for accurate binary segmentation, powered by `segmentation_models_pytorch`.
- **FastAPI Backend:** High-performance async backend with Swagger docs, health checks, and CORS enabled.
- **Memory-Efficient Processing:** The inference pipeline operates entirely in memory, returning Base64 data URIs. No disk I/O overhead for saving output images, resulting in faster and cleaner execution.
- **Dynamic Web UI:** A simple, drag-and-drop web UI to test and visualize segmentation masks and overlays seamlessly.

---

## Architecture & Pipeline

### Model Specifications

- **Framework:** PyTorch & `segmentation_models_pytorch`
- **Architecture:** U-Net
- **Encoder:** EfficientNet-B7 (Pretrained on ImageNet)
- **Decoder Attention:** scSE (Spatial and Channel Squeeze & Excitation)
- **Input Dimensions:** RGB, resized to 256×256
- **Normalization:** ImageNet standards (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`)
- **Output:** Single-channel binary mask (with internal Sigmoid activation, thresholded at 0.5)

### Inference Workflow

1. **Upload & Validation:** API accepts `jpg`, `jpeg`, `png`, `tif`, `tiff` via `multipart/form-data`.
2. **Preprocessing:** Converts grayscale/RGBA to RGB, resizes to 256×256, and applies ImageNet normalization.
3. **Prediction:** Tensor is passed through the U-Net model.
4. **Postprocessing:** Predictions are thresholded at 0.5. Masks are resized back to the original image dimensions. An overlay heatmap is generated for visualization.
5. **Base64 Encoding:** The original, mask, and overlay images are encoded to Base64 data URIs and returned directly as a JSON response.

---

## Project Structure

```text
Brain-MRI-Segmentation/
├── requirements.txt
├── README.md
├── LICENSE
├── mri/                      # Optional Python virtual environment
└── src/
    ├── main.py               # FastAPI application entry point
    ├── .env.example          # Environment variables template
    ├── assets/               # Temporary storage for uploaded files
    ├── controllers/
    │   ├── BaseController.py # Base logic for controllers
    │   └── DataController.py # Validation and file handling logic
    ├── models/
    │   ├── enums/            # System responses and enumerations
    │   └── Model/
    │       ├── unet.py       # U-Net + EfficientNet-B7 architecture definition
    │       ├── inference.py  # Preprocessing, prediction, and Base64 encoding
    │       └── best_model.pth # PyTorch model weights (requires manual download)
    ├── Notebook/             # Jupyter notebook for training references
    ├── routes/
    │   ├── base.py           # Base API routing
    │   ├── data.py           # Segmentation endpoints
    │   └── schemas/
    │       └── DataSchema.py # Pydantic models for API responses
    ├── static/               # CSS and JS for the frontend
    ├── templates/            # HTML templates for the UI
    └── utils/                # Configuration and utilities
```

---

## Setup & Installation (Windows)

1. **Clone the repository and enter the directory:**
   ```powershell
   git clone <repository_url>
   cd Brain-MRI-Segmentation
   ```

2. **Activate the virtual environment:**
   ```powershell
   .\mri\Scripts\Activate.ps1
   ```
   *(Or create your own: `python -m venv venv`)*

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Download Model Weights:**
   Place the trained `best_model.pth` inside `src/models/Model/`.
   *Note: You can download the weights from [this Kaggle Notebook](https://www.kaggle.com/code/mahmoudabdulghany/brain-mri-segmentation-from-clahe-to-93-9-dice/output).*

5. **Start the API:**
   ```powershell
   cd src
   python -m uvicorn main:app --reload
   ```

6. **Access the application:**
   - **UI:** http://127.0.0.1:8000/
   - **Swagger Docs:** http://127.0.0.1:8000/docs
   - **Health Check:** http://127.0.0.1:8000/health

---

## API Reference

### Segment Image

- **Endpoint:** `/api/v1/segment`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Payload:** `file` (Image file)

#### Response Schema (JSON)

Returns a JSON object containing Base64 encoded images that can be embedded directly into HTML using data URIs.

```json
{
  "filename": "scan_001",
  "original_image_url": "data:image/png;base64,iVBORw0KGgo...",
  "mask_url": "data:image/png;base64,iVBORw0KGgo...",
  "overlay_url": "data:image/png;base64,iVBORw0KGgo...",
  "width": 512,
  "height": 512,
  "has_tumor": true
}
```

### cURL Example

```bash
curl -X POST \
  -F "file=@C:/path/to/your/mri.tif" \
  http://127.0.0.1:8000/api/v1/segment
```

---

## Troubleshooting

- **Missing Weights Error:** Ensure `best_model.pth` is physically present in `src/models/Model/`.
- **CUDA/GPU Issues:** If CUDA is unavailable, the application automatically falls back to CPU inference. Ensure PyTorch is installed with proper CUDA support for GPU acceleration if desired.
- **500 Internal Server Error on Upload:** Check if the file exceeds size limits or is an unsupported format. Confirm that the application has write permissions for the temporary upload directory (`src/assets/files`).

---

## Notes

- This repository includes a project-specific `mri/` virtual environment folder for convenience.
- The original Jupyter notebook used for dataset preparation and training is located at `src/Notebook/brain-mri.ipynb`.
