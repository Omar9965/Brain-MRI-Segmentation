import torch
import os
import cv2
import base64
import numpy as np
from typing import Optional, Union, Dict, Any
from .unet import model as segmentation_model


# Paths
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization constants (matching EfficientNet-B7 pretrained encoder)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Global model instance (lazy loading)
_model = None


def load_model(model_path: str = BEST_MODEL_PATH):
    """Load the segmentation model with trained weights."""
    global _model
    if _model is None:
        _model = segmentation_model
        _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model


def preprocess_image(image: np.ndarray) -> tuple:
    """Preprocess image for model inference.
    
    Pipeline:
        1. Convert to RGB
        2. Store original size
        3. Resize to 256×256
        4. Normalize to [0,1] then apply ImageNet mean/std
        5. Convert HWC → CHW tensor
    """
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = (image.shape[1], image.shape[0])
    
    # Resize to expected input size (256x256)
    image = cv2.resize(image, (256, 256))
    
    # Normalize: scale to [0,1] then apply ImageNet mean/std
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image, original_size


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create an overlay of the mask on the original image."""
    # Ensure mask is same size as original
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Create colored mask (red for tumor)
    colored_mask = np.zeros_like(original)
    colored_mask[:, :, 2] = mask_resized  # Red channel
    
    # Blend
    overlay = cv2.addWeighted(original, 1, colored_mask, alpha, 0)
    return overlay


def segment_image(
    image: Union[str, np.ndarray],
    filename: str = "image",
    model=None,
    return_overlay: bool = True
) -> Dict[str, Any]:
    """
    Segment a single brain MRI image.
    
    Returns base64-encoded images (no files saved to disk).
    The images persist in memory only while the user is on the page.
    
    Args:
        image: Either a file path (str) or numpy array (BGR format)
        filename: Original filename (without extension) for display
        model: Optional pre-loaded model instance
        return_overlay: Whether to include overlay visualization
    
    Returns:
        Dict with base64 data URIs for images, dimensions, and tumor detection status
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load image if path is provided
    if isinstance(image, str):
        original_image = cv2.imread(image)
        if original_image is None:
            raise ValueError(f"Could not load image: {image}")
    else:
        original_image = image.copy()
    
    original_h, original_w = original_image.shape[:2]
    
    # Preprocess
    input_tensor, _ = preprocess_image(original_image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Inference — model already applies sigmoid internally
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output > 0.5).float()
    
    # Convert mask to numpy
    mask_np = pred_mask.squeeze().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # Resize mask back to original size
    mask_resized = cv2.resize(mask_np, (original_w, original_h))
    
    # Check if tumor is detected
    has_tumor = bool(np.any(mask_resized > 0))
    
    # Encode images as base64 data URIs (no disk writes)
    original_b64 = f"data:image/png;base64,{numpy_to_base64(original_image)}"
    mask_b64 = f"data:image/png;base64,{numpy_to_base64(mask_resized)}"
    
    # Create and encode overlay if requested
    overlay_b64 = None
    if return_overlay:
        overlay = create_overlay(original_image, mask_resized)
        overlay_b64 = f"data:image/png;base64,{numpy_to_base64(overlay)}"
    
    return {
        "filename": filename,
        "original_image_url": original_b64,
        "mask_url": mask_b64,
        "overlay_url": overlay_b64,
        "width": original_w,
        "height": original_h,
        "has_tumor": has_tumor
    }
