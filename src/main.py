from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import data_router, base_router
import os

app = FastAPI(
    title="Brain MRI Tumor Segmentation API",
    description="U-Net based brain MRI tumor segmentation with CBAM attention",
    version="0.1.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
output_dir = os.path.join(os.path.dirname(__file__), "assets", "output")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# Serve the main page
@app.get("/")
async def serve_homepage():
    return FileResponse(os.path.join(templates_dir, "index.html"))

app.include_router(base_router)
app.include_router(data_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


