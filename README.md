# Brain MRI Tumor Segmentation

A FastAPI + PyTorch application for brain tumor segmentation on MRI scans using a U-Net model enhanced with CBAM attention. It provides a simple web UI and a clean REST API for single or batch image inference.

---

## Features

- U-Net with CBAM attention for accurate binary segmentation
- FastAPI backend with Swagger docs and CORS enabled
- Saves results (original, mask, overlay) to disk and returns public URLs
- Simple drag-and-drop web UI
- Batch processing endpoint

---

## Project Structure

```
requirements.txt
mri/                      # Python venv (local)
src/
  main.py                 # FastAPI app entry
  assets/
    files/               # Temp uploads
    output/              # Saved outputs (served at /output)
  controllers/
  models/
    Model/
      unet.py            # U-Net + CBAM
      inference.py       # Inference utilities
      best_model.pth     # Place weights here
  routes/
    data.py              # API routes
    schemas/DataSchema.py
  static/                # JS/CSS for UI
  templates/             # index.html
  utils/
```

---

## Model

- Architecture: U-Net with CBAM (Channel + Spatial attention)
- Input: RGB, resized to 256×256, normalized with mean=0.5, std=0.5
- Output: Single-channel mask (sigmoid + 0.5 threshold)
- Weights: Place `best_model.pth` in `src/models/Model/`

Preprocessing (inference):
- Converts BGR/GRAY to RGB
- Resizes to 256×256
- Normalizes to `[-1, 1]` via `(img/255-0.5)/0.5`

---

## Run Locally (Windows)

```powershell
# 1) Activate virtual environment (provided)
.\mri\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start the API
cd src
uvicorn main:app --reload
```

- UI: http://127.0.0.1:8000/
- Docs (Swagger): http://127.0.0.1:8000/docs
- Outputs served at: http://127.0.0.1:8000/output/

---

## API

### Endpoints

- POST `/api/v1/segment` — Segment a single MRI image
- POST `/api/v1/segment-multiple` — Segment multiple MRI images

Accepted types: JPG/JPEG, PNG, TIFF (`.tif`, `.tiff`). Max size defaults to 500MB in the UI; server-side limits may differ.

### Response Schema (single)

```json
{
  "filename": "scan_001",
  "original_image_url": "/output/scan_001_1702828800000_original.png",
  "mask_url": "/output/scan_001_1702828800000_mask.png",
  "overlay_url": "/output/scan_001_1702828800000_overlay.png",
  "width": 512,
  "height": 512,
  "has_tumor": true
}
```

`overlay_url` may be null if overlays are disabled.

### cURL Examples

Single image:
```bash
curl -X POST \
  -F "file=@C:/path/to/your/mri.tif" \
  http://127.0.0.1:8000/api/v1/segment
```

Multiple images:
```bash
curl -X POST \
  -F "files=@C:/path/scan1.png" \
  -F "files=@C:/path/scan2.tif" \
  http://127.0.0.1:8000/api/v1/segment-multiple
```

---

## Frontend

- Served at `/` with `index.html`
- Drag-and-drop or file picker uploads
- Displays Original, Mask, and Overlay using API-returned URLs

---

## Storage & Paths

- Temp uploads: `src/assets/files/` (cleaned after processing)
- Saved results: `src/assets/output/` and exposed at `/output` via FastAPI static mount
- File naming: `{original_filename}_{timestamp}_{type}.png`

---

## Configuration

- Model weights: `src/models/Model/best_model.pth`
- Device: CUDA if available, else CPU
- Threshold: 0.5 on sigmoid output

---

## Troubleshooting

- Missing weights: Ensure `best_model.pth` exists in `src/models/Model/` you can download it from here "https://www.kaggle.com/code/omarmohamed89/brain-mri/output"
- 500 errors on upload: Check file type/size and server logs
- Images not visible: Confirm `/output` is mounted and URLs resolve
- CUDA issues: App will automatically fall back to CPU

---

## Notes

- This repo includes a project-specific venv folder (`mri/`) for convenience; you may use your own environment instead.
- The included Jupyter notebook for training is at `src/Notebook/brain-mri.ipynb` (dataset and training code reference).
# Brain-MRI-Segmentation
