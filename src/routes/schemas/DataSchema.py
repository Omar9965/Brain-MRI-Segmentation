from pydantic import BaseModel
from typing import List, Optional

class SegmentationResult(BaseModel):
    """Result for a single brain MRI segmentation"""
    filename: str                 
    original_image_url: str        
    mask_url: str                 
    overlay_url: Optional[str]     
    width: int
    height: int
    has_tumor: bool              


class MultipleSegmentationResponse(BaseModel):
    """Response for multiple MRI segmentation"""
    results: List[SegmentationResult]