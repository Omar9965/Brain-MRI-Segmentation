from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import os
import shutil
import uuid
import time
from controllers import DataController
from routes.schemas.DataSchema import SegmentationResult, MultipleSegmentationResponse
from models import segment_image, segment_multiple_images
from utils.async_processor import BatchProcessor, TaskPriority
from utils.websocket_manager import manager
import asyncio
import json
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["brain-mri-segmentation"])

# Initialize controller
data_controller = DataController()

# Global async processor
async_processor = None

async def get_async_processor():
    """Get or create the global async processor"""
    global async_processor
    if async_processor is None:
        async_processor = BatchProcessor(
            max_concurrent_tasks=4,
            progress_callback=manager.send_progress_update
        )
    return async_processor


@router.post("/segment", response_model=SegmentationResult)
async def segment_single_mri(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
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
        
        # Process the image
        result = segment_image(temp_path, base_filename)
        
        # Broadcast completion
        await manager.send_completion(base_filename, result)
        
        return result
        
    except Exception as e:
        await manager.send_error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/segment-multiple", response_model=MultipleSegmentationResponse)
async def segment_multiple_mri(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload multiple brain MRI images and perform tumor segmentation.
    
    Args:
        files: List of image files (jpg, png, tiff, jpeg, tif)
        
    Returns:
        MultipleSegmentationResponse with list of segmentation results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files
    is_valid, message = await data_controller.validate_images(files)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Create temporary directory for batch processing
    temp_dir = os.path.join(data_controller.file_dir, f"batch_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save all files
        file_paths = []
        for file in files:
            original_filename = data_controller.get_filename(file.filename)
            base_filename = os.path.splitext(original_filename)[0]
            
            random_string = data_controller.generate_random_string()
            ext = file.filename.split('.')[-1].lower()
            temp_filename = f"{random_string}.{ext}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append((temp_path, base_filename))
        
        # Process files using async processor
        processor = await get_async_processor()
        task_ids = []
        
        for file_path, filename in file_paths:
            task_id = await processor.add_task(filename, file_path, TaskPriority.NORMAL)
            task_ids.append(task_id)
        
        # Return immediate response with task IDs
        return {
            "session_id": str(uuid.uuid4()),
            "task_ids": task_ids,
            "message": f"Processing {len(files)} images. Use WebSocket endpoint for real-time updates."
        }
        
    except Exception as e:
        await manager.send_error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


@router.get("/segment-status/{task_id}")
async def get_segmentation_status(task_id: str):
    """
    Get the status of a specific segmentation task.
    
    Args:
        task_id: The task ID to check status for
        
    Returns:
        Task status and progress information
    """
    processor = await get_async_processor()
    task = await processor.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task.task_id,
        "filename": task.filename,
        "status": task.status.value,
        "progress": task.progress,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error": task.error
    }


@router.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a specific segmentation task.
    
    Args:
        task_id: The task ID to cancel
        
    Returns:
        Success status and task information
    """
    processor = await get_async_processor()
    success = await processor.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return {"success": True, "task_id": task_id, "message": "Task cancelled successfully"}


@router.get("/batch-status")
async def get_batch_status():
    """
    Get the status of all batch processing tasks.
    
    Returns:
        Complete status of all tasks in the processor
    """
    processor = await get_async_processor()
    tasks = await processor.get_all_tasks()
    
    return {
        "total_tasks": len(tasks),
        "pending_tasks": len([t for t in tasks if t.status.value == "pending"]),
        "processing_tasks": len([t for t in tasks if t.status.value == "processing"]),
        "completed_tasks": len([t for t in tasks if t.status.value == "completed"]),
        "failed_tasks": len([t for t in tasks if t.status.value == "failed"]),
        "cancelled_tasks": len([t for t in tasks if t.status.value == "cancelled"]),
        "tasks": [
            {
                "task_id": task.task_id,
                "filename": task.filename,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error": task.error
            }
            for task in tasks
        ]
    }
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