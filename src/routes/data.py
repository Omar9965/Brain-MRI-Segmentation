from fastapi import APIRouter, UploadFile, File, HTTPException 
import os
import shutil
from controllers import DataController
from routes.schemas.DataSchema import SegmentationResult
from models import segment_image

router = APIRouter(prefix="/api/v1", tags=["brain-mri-segmentation"])

# Initialize controller
data_controller = DataController()

@router.post("/segment", response_model=SegmentationResult)
async def segment_single_mri(
    file: UploadFile = File(...),
):
    """
    Upload a single brain MRI image and perform tumor segmentation.
    
    Args:
        file: Single image file (jpg, png, tiff, jpeg, tif)
        
    Returns:
        SegmentationResult with URLs to original image, mask, overlay, and tumor detection status
    """
    is_valid, message = await data_controller.validate_images([file])
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    temp_path = None
    
    try:
        # Ensure upload directory exists
        os.makedirs(data_controller.file_dir, exist_ok=True)
        
        # Get original filename without extension
        base_filename = os.path.splitext(file.filename)[0]
        
        # Save uploaded file temporarily
        random_string = data_controller.generate_random_string()
        ext = file.filename.split('.')[-1].lower()
        temp_filename = f"{random_string}.{ext}"
        temp_path = os.path.join(data_controller.file_dir, temp_filename)
        
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = segment_image(temp_path, base_filename)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)