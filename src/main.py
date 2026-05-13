from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import data_router, base_router
import os
import logging
from datetime import datetime, timezone
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain MRI Tumor Segmentation API",
    description="U-Net based brain MRI tumor segmentation with CBAM attention - Enhanced with async processing and WebSocket support",
    version="1.0.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
output_dir = os.path.join(os.path.dirname(__file__), "assets", "output")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# Add CORS middleware before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the main page
@app.get("/")
async def serve_homepage():
    return FileResponse(os.path.join(templates_dir, "index.html"))

app.include_router(base_router)
app.include_router(data_router)

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    logger.info("Starting Brain MRI Segmentation API...")
    logger.info("WebSocket endpoints available at /ws/progress/{session_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Brain MRI Segmentation API...")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


