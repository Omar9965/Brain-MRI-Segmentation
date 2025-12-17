from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from controllers import DataController
from routes.schemas.DataSchema import SegmentationResult, MultipleSegmentationResponse
from models import segment_image, segment_multiple_images

router = APIRouter(prefix="/api/v1", tags=["brain-mri-segmentation"])

# Initialize controller
data_controller = DataController()


@router.post("/segment", response_model=SegmentationResult)
async def segment_single_mri(file: UploadFile = File(...)):
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
        original_filename = data_controller.get_filename(file.filename)
        base_filename = os.path.splitext(original_filename)[0]
        
        # Save uploaded file temporarily
        random_string = data_controller.generate_random_string()
        ext = file.filename.split('.')[-1].lower()
        temp_filename = f"{random_string}.{ext}"
        temp_path = os.path.join(data_controller.file_dir, temp_filename)
        
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run segmentation with original filename
        result = segment_image(temp_path, filename=base_filename, return_overlay=True)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Segmentation failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Failed to remove temp file {temp_path}: {str(e)}")


@router.post("/segment-multiple", response_model=MultipleSegmentationResponse)
async def segment_multiple_mris(files: List[UploadFile] = File(...)):
    """
    Upload multiple brain MRI images and perform tumor segmentation on all.
    
    Args:
        files: List of image files (jpg, png, tiff, jpeg)
        
    Returns:
        MultipleSegmentationResponse with URLs for all segmented images
    """
    is_valid, message = await data_controller.validate_images(files)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    temp_paths = []
    images_with_filenames = []
    
    try:
        # Ensure upload directory exists
        os.makedirs(data_controller.file_dir, exist_ok=True)
        
        # Save uploaded files temporarily
        for file in files:
            # Get original filename without extension
            original_filename = data_controller.get_filename(file.filename)
            base_filename = os.path.splitext(original_filename)[0]
            
            random_string = data_controller.generate_random_string()
            ext = file.filename.split('.')[-1].lower()
            temp_filename = f"{random_string}.{ext}"
            temp_path = os.path.join(data_controller.file_dir, temp_filename)
            
            # Save file
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            temp_paths.append(temp_path)
            images_with_filenames.append((temp_path, base_filename))
        
        # Run segmentation on all uploaded images with filenames
        result = segment_multiple_images(images_with_filenames, return_overlay=True)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Segmentation failed: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Failed to remove temp file {temp_path}: {str(e)}")